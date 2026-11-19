"""Economic parameters and provider routes."""

from dataclasses import dataclass
import math

MODELS = {
    "gpt-4o": ("openai", "gpt-4o"),
    "gpt-4o-mini": ("openai", "gpt-4o-mini"),
    "llama-3.3-70b": ("groq", "llama-3.3-70b-versatile"),
    "llama-4-scout-17b": ("groq", "meta-llama/llama-4-scout-17b-16e-instruct"),
    "qwen3-32b": ("groq", "qwen/qwen3-32b"),
    "deepseek-r1": ("groq", "deepseek-r1-distill-llama-70b"),
    "claude-sonnet-4": ("anthropic", "claude-sonnet-4-20250514"),
}
CONDITIONS = ("all", "without_social_learning", "without_group_norm", "neither")
ABLATIONS = {"all": [], "without_social_learning": ["social learning"],
             "without_group_norm": ["group norm"], "neither": ["social learning", "group norm"]}


@dataclass(frozen=True)
class Parameters:
    n_agents: int = 10
    init_resource: float = 300
    capacity: float = 300
    growth_rate: float = 0.6
    harvest_productivity: float = 0.05
    penalty: float = 10
    punish_cost: float = 0
    consumption: float = 1.0
    altruism_ratio: float = 0.5
    monitoring_cost: float = 0.0
    punish_p_min: float = 0.7

    def __post_init__(self):
        if type(self.n_agents) is not int or self.n_agents < 2:
            raise ValueError("At least two agents are required")
        for name, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid parameter: {name}")
        if self.capacity <= 0 or self.init_resource > self.capacity:
            raise ValueError("Initial resource must not exceed positive capacity")
        if not 0 <= self.altruism_ratio <= 1 or not 0 <= self.punish_p_min <= 0.7:
            raise ValueError("Invalid initial composition or punishment probability")
