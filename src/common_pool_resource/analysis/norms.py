"""Embedding-based similarity of personal and collective norms."""

import numpy as np
import pandas as pd

from .tables import GROUPS, load
from ..data import dumps, empty_output


def metrics(personal, group):
    vectors = np.asarray(personal, dtype=float)
    if vectors.ndim != 2 or not len(vectors):
        raise ValueError("Expected a nonempty matrix of personal-norm embeddings")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or np.any(norms == 0):
        raise ValueError("Invalid personal-norm embedding")
    vectors = vectors / norms
    gram = vectors @ vectors.T
    np.fill_diagonal(gram, 0)
    result = {"personal_similarity_zero_diagonal": float(gram.mean()),
              "personal_similarity_pairwise": float(gram.sum() / (len(gram) * (len(gram)-1))) if len(gram) > 1 else np.nan}
    if group is not None:
        group = np.asarray(group, dtype=float)
        if not np.isfinite(group).all() or np.linalg.norm(group) == 0:
            raise ValueError("Invalid group-norm embedding")
        result["personal_group_alignment"] = float(np.mean(vectors @ (group / np.linalg.norm(group))))
    return result


def run(data, output, encoder, trials=10, encode=None):
    empty_output(output, data)
    frame, receipt = load(data, trials=trials)
    if not {"individual_norm", "group_norm"} <= set(frame):
        raise ValueError("Norm analysis requires LLM outputs with personal and group statements")
    if encode is None:
        if not encoder.is_dir():
            raise ValueError("Supply a local mixedbread-ai/mxbai-embed-xsmall-v1 model directory")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(str(encoder.resolve()), device="cpu", local_files_only=True)
        encode = lambda texts: model.encode(texts, batch_size=32, normalize_embeddings=True)
    keys = [*GROUPS, "file_id", "RunId", "Step"]
    texts = sorted(set(frame.loc[frame.AgentID.notna(), "individual_norm"]) |
                   {s for s in frame.group_norm if isinstance(s, str) and s and s != "N/A"})
    vectors = np.asarray(encode(texts))
    if vectors.ndim != 2 or len(vectors) != len(texts) or not np.isfinite(vectors).all():
        raise ValueError("Encoder returned invalid embeddings")
    embedding = dict(zip(texts, vectors))
    rows = []
    for values, group in frame[frame.AgentID.notna()].groupby(keys, sort=True):
        shared = group.group_norm.unique()
        if len(shared) != 1:
            raise ValueError("Multiple group norms in one society step")
        result = metrics([embedding[s] for s in group.individual_norm], embedding.get(shared[0]))
        rows.append({**dict(zip(keys, values)), **result, "group_norm": shared[0]})
    table = pd.DataFrame(rows)
    table.to_csv(output / "norm_metrics.csv", index=False)
    previous = frame[keys + ["group_norm"]].drop_duplicates().sort_values(keys)
    transitions = []
    for _, society in previous.groupby(keys[:-1], sort=True):
        last = None
        for row in society.to_dict("records"):
            current = embedding.get(row["group_norm"])
            if current is not None and np.linalg.norm(current) == 0:
                raise ValueError("Invalid group-norm embedding")
            if current is not None and last is not None:
                similarity = float(np.dot(current, last) / (np.linalg.norm(current) * np.linalg.norm(last)))
                transitions.append({**row, "group_norm_stability": similarity})
            last = current
    pd.DataFrame(transitions, columns=keys + ["group_norm", "group_norm_stability"]).to_csv(
        output / "group_norm_transitions.csv", index=False)
    parts = [table.melt(id_vars=keys, value_vars=[c for c in table if c.startswith("personal_")],
                        var_name="metric", value_name="value")]
    if transitions:
        parts.append(pd.DataFrame(transitions).rename(columns={"group_norm_stability": "value"}).assign(metric="group_norm_stability"))
    paths = pd.concat(parts).groupby(GROUPS + ["Step", "metric"]).value.agg(["mean", "sem", "count"]).reset_index()
    paths.to_csv(output / "norm_trajectories.csv", index=False)
    receipt.update(encoder=str(encoder), metrics="zero-diagonal population mean and off-diagonal pair mean reported separately",
                   scope="within-export-row personal/group comparisons; published CSV offset is retained",
                   transitions="one comparison per adjacent group-norm pair; no embedding-batch boundary duplication")
    (output / "analysis.json").write_text(dumps(receipt) + "\n")
    return receipt
