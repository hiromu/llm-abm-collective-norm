"""Numeric-masked sentence embeddings and society-seed held-out decoding."""

from pathlib import Path
import hashlib
import json
import re
import warnings

import numpy as np

from ..config import ARMS, DECISION_ROUNDS, MARKETS, MODELS, QUALITIES, SEEDS
from ..data import prepare_output, read_group, verify, write_csv

NUMERIC = re.compile(r"(?<!\w)[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?%?(?!\w)")


def mask_numbers(text: str) -> str:
    return NUMERIC.sub("number", text)


def normalize(values):
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


class Encoder:
    """all-MiniLM-L6-v2, CPU ONNX, token-weighted normalized chunk pooling."""

    def __init__(self, model_dir: Path, threads=4):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(model_dir / "onnx/model.onnx"),
                                           sess_options=options, providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.input_names = {i.name for i in self.session.get_inputs()}

    def encode(self, texts, batch_size=48):
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        unique = list(dict.fromkeys(texts))
        index = {s: i for i, s in enumerate(unique)}
        tokens = self.tokenizer.encode_batch(unique, add_special_tokens=False)
        chunks = []
        for i, item in enumerate(tokens):
            for start in range(0, max(1, len(item.ids)), 254):
                part = item.ids[start:start + 254]
                chunks.append((i, len(part), [101] + part + [102]))
        chunks.sort(key=lambda x: len(x[2]))
        out = np.zeros((len(unique), 384), dtype=np.float32)
        weights = np.zeros(len(unique), dtype=np.float32)
        for start in range(0, len(chunks), batch_size):
            part = chunks[start:start + batch_size]
            width = max(len(x[2]) for x in part)
            ids = np.zeros((len(part), width), dtype=np.int64)
            mask = np.zeros_like(ids)
            for j, (_, _, values) in enumerate(part):
                ids[j, :len(values)] = values
                mask[j, :len(values)] = 1
            feeds = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
            result = self.session.run(None, {k: v for k, v in feeds.items() if k in self.input_names})[0]
            if result.ndim == 3:
                result = (result * mask[:, :, None]).sum(1) / np.maximum(mask.sum(1, keepdims=True), 1)
            if result.shape != (len(part), 384):
                raise ValueError("Unexpected encoder output shape")
            result = normalize(result)
            for j, (owner, n, _) in enumerate(part):
                out[owner] += result[j] * max(n, 1)
                weights[owner] += max(n, 1)
        out = normalize(out / weights[:, None])
        if not np.isfinite(out).all():
            raise ValueError("Non-finite sentence embedding")
        return out[[index[s] for s in texts]]


def verify_encoder(data: Path, model_dir: Path):
    hashes = json.loads((data / "encoder_files.json").read_text())
    for name, expected in hashes.items():
        if hashlib.sha256((model_dir / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Encoder file differs from the pinned revision: {name}")


def seed_folds(rows, market=None):
    seeds = np.array([r["seed"] for r in rows])
    selected = np.array([market is None or r["market"] == market for r in rows])
    for seed in SEEDS:
        train, test = selected & (seeds != seed), selected & (seeds == seed)
        if train.sum() != (1600 if market is None else 800) or test.sum() != (400 if market is None else 200):
            raise ValueError("Decoding requires the complete balanced five-seed cohort")
        yield seed, train, test


def decode(values, rows):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    market_labels = np.array([MARKETS.index(r["market"]) for r in rows])
    quality_labels = np.array([QUALITIES.index(r["quality"]) for r in rows])
    identity = {k: rows[0][k] for k in ("model", "arm", "round")}
    market_scores, quality_scores = [], []
    for context in (None, *MARKETS):
        target = market_labels if context is None else quality_labels
        for seed, train, test in seed_folds(rows, context):
            classifier = LogisticRegression(C=1, max_iter=1000, random_state=6411)
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                classifier.fit(values[train], target[train])
            prediction = classifier.predict(values[test])
            result = {**identity, "held_out_seed": seed, "n_test": int(test.sum()),
                      "accuracy": float((prediction == target[test]).mean())}
            if context is None:
                market_scores.append(result)
            else:
                cm = confusion_matrix(target[test], prediction, labels=[0, 1])
                result.update(market=context, HQ_recall=float(cm[0, 0] / 100),
                              LQ_recall=float(cm[1, 1] / 100), HQ_as_HQ=int(cm[0, 0]),
                              HQ_as_LQ=int(cm[0, 1]), LQ_as_HQ=int(cm[1, 0]), LQ_as_LQ=int(cm[1, 1]))
                quality_scores.append(result)
    return market_scores, quality_scores


def analyze(data: Path, output: Path, encoder: Encoder, models=MODELS, progress=None):
    """Encode strategy statements and fit decoders and PCA projections.

    PCA is fit separately to all 132,000 statements of each model. Labels are
    used only for the decoder targets and for aggregating the PCA coordinates.
    """
    from sklearn.decomposition import PCA
    from threadpoolctl import threadpool_limits
    market_scores, quality_scores, projected = [], [], []
    with threadpool_limits(limits=4):
        for model in models:
            values = np.empty((132000, 384), dtype=np.float32)
            centers, metadata = [], []
            offset = 0
            for arm in ARMS:
                for round_ in DECISION_ROUNDS:
                    rows = read_group(data, model, arm, round_)
                    group_values = encoder.encode([mask_numbers(r["strategy"]) for r in rows])
                    values[offset:offset + len(rows)] = group_values
                    offset += len(rows)
                    a, b = decode(group_values, rows)
                    market_scores.extend(a)
                    quality_scores.extend(b)
                    for market in MARKETS:
                        for seed in SEEDS:
                            for quality in QUALITIES:
                                selected = np.array([r["market"] == market and r["seed"] == seed
                                                     and r["quality"] == quality for r in rows])
                                if selected.sum() != 100:
                                    raise ValueError("Unbalanced quality subgroup")
                                subset = group_values[selected].astype(np.float64)
                                center = subset.mean(0)
                                total = subset.sum(0)
                                dispersion = 1 - float((total @ total - np.square(subset).sum()) / (100 * 99))
                                metadata.append({"model": model, "arm": arm, "round": round_,
                                                 "market": market, "seed": seed, "quality": quality,
                                                 "n": 100, "within_dispersion": dispersion,
                                                 "vector_row": len(centers)})
                                centers.append(center)
                    if progress:
                        progress(f"Encoded and decoded {model}/{arm}/round {round_}")
            pca = PCA(n_components=2, svd_solver="covariance_eigh").fit(values)
            centers = np.asarray(centers)
            xy = pca.transform(centers)
            np.savez_compressed(output / f"{model}.npz", centroids=centers,
                                components=pca.components_, mean=pca.mean_,
                                explained_variance_ratio=pca.explained_variance_ratio_)
            projected.extend({**row, "pc1": float(point[0]), "pc2": float(point[1])}
                             for row, point in zip(metadata, xy))
            del values
    write_csv(output / "market_decoding.csv", market_scores)
    write_csv(output / "quality_decoding.csv", quality_scores)
    write_csv(output / "centroid_projections.csv", projected)
    return {"models": list(models), "statements": 132000 * len(models),
            "market_folds": len(market_scores), "quality_folds": len(quality_scores)}


def run(data, output, encoder):
    receipt = {"inputs": verify(data, scope="analysis")}
    verify_encoder(data, encoder)
    prepare_output(output, data, encoder)
    receipt["analysis"] = analyze(data, output, Encoder(encoder),
                                  progress=lambda s: print(s, flush=True))
    return receipt
