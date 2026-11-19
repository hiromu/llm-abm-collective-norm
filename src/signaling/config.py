"""Population, market and model configurations."""

MODELS = ("deepseek", "luna", "oss", "qwen")
LABELS = {
    "deepseek": "DeepSeek-4.1 Flash",
    "luna": "GPT-6 Luna",
    "oss": "GPT-OSS 120B",
    "qwen": "Qwen-3.5 35B",
}
ARMS = ("individual", "social")
MARKETS = ("40/10", "10/10")
QUALITIES = ("HQ", "LQ")
SEEDS = tuple(range(332140, 332145))
DECISION_ROUNDS = tuple(range(4, 101, 3))
ROUNDS = 100
