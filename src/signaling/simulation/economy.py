"""Continuous technology choice, competitive hiring and synchronous learning."""

from dataclasses import dataclass
import math
from numbers import Real
import random


@dataclass(frozen=True)
class Parameters:
    applicants: int = 200
    employers: int = 20
    positions: int = 5
    high_quality_fraction: float = 0.5
    high_productivity: float = 40.0
    low_productivity: float = 10.0
    applicant_cost: float = 1.0
    employer_cost: float = 2.0
    high_weight: float = 0.8
    low_weight: float = 0.3
    recognition_loss: float = 0.2
    technology_penalty: float = 0.15
    learning_interval: int = 3
    selection_strength: float = 10.0
    employer_selection_strength: float = 10.0
    initial_profile: str = "centered"
    initial_high: float = 0.5
    initial_low: float = 0.5
    initial_employer: float = 0.5
    initial_spread: float = 1.0
    initial_majority_share: float = 0.9
    payoff_memory: str = "window"
    applicant_cost_basis: str = "submission"
    employer_cost_basis: str = "round"

    def __post_init__(self):
        for name in ("applicants", "employers", "positions", "learning_interval"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        bounded = ("high_quality_fraction", "high_weight", "low_weight", "recognition_loss",
                   "technology_penalty", "initial_high", "initial_low", "initial_employer",
                   "initial_spread", "initial_majority_share")
        nonnegative = ("high_productivity", "low_productivity", "applicant_cost", "employer_cost")
        positive = ("selection_strength", "employer_selection_strength")
        for name in (*bounded, *nonnegative, *positive):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            if (name in bounded and not 0 <= value <= 1
                    or name in nonnegative and value < 0 or name in positive and value <= 0):
                raise ValueError(f"{name} is outside its allowed range")
        if self.high_productivity < self.low_productivity:
            raise ValueError("High-quality productivity must be at least low-quality productivity")
        if self.high_weight <= self.low_weight:
            raise ValueError("High-quality hiring weight must exceed low-quality hiring weight")
        for name, allowed in {
            "initial_profile": ("centered", "high_mixture", "low_mixture"),
            "payoff_memory": ("window", "lifetime"),
            "applicant_cost_basis": ("submission", "hired"),
            "employer_cost_basis": ("round", "hire"),
        }.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}")

    @classmethod
    def for_market(cls, market: str):
        try:
            high, low = map(float, market.split("/"))
        except (AttributeError, ValueError) as error:
            raise ValueError("Market must be a productivity pair such as 40/10") from error
        return cls(high_productivity=high, low_productivity=low)


@dataclass(eq=False)
class Agent:
    index: int
    quality: str
    technology: float = 0.0
    used: float = 0.0
    gross: float = 0.0
    cost: float = 0.0
    net: float = 0.0
    window: float = 0.0
    cumulative: float = 0.0
    hired: bool = False


def hiring_weight(quality: str, technology: float, employer_technology: float,
                  p: Parameters) -> float:
    technology = max(0.0, min(1.0, technology))
    employer_technology = max(0.0, min(1.0, employer_technology))
    weight = p.high_weight if quality == "HQ" else p.low_weight
    if quality == "HQ":
        weight -= employer_technology * p.recognition_loss
    weight -= technology * (1.0 - employer_technology) * p.technology_penalty
    return max(0.0, min(1.0, weight))


def imitation_probability(own: float, peer: float, strength: float) -> float:
    x = max(-60.0, min(60.0, strength * (peer - own)))
    return 1.0 / (1.0 + math.exp(-x))


class Market:
    """Continuous job-market model with synchronous learning barriers.

    Both information arms consume the same kinds of peer and imitation draws.
    Employers always apply the payoff-based Fermi rule. The applicant Fermi
    proposal is used only when ``policy='abm'``. Pass a productivity pair or
    a ``Parameters`` instance to use the same engine for ABM and LLM experiments.
    """

    def __init__(self, market: str | Parameters, seed: int, information: str = "social",
                 policy: str = "llm", rounds: int = 100):
        if information not in ("individual", "social") or policy not in ("llm", "abm"):
            raise ValueError("Invalid information condition or policy")
        if type(rounds) is not int or rounds < 1:
            raise ValueError("rounds must be a positive integer")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        self.p = market if isinstance(market, Parameters) else Parameters.for_market(market)
        self.rng = random.Random(seed)
        self.information, self.policy, self.rounds = information, policy, rounds
        self.round = 0
        self.pending = None
        n_high = int(math.floor(self.p.high_quality_fraction * self.p.applicants + 0.5))
        quality = ["HQ"] * n_high + ["LQ"] * (self.p.applicants - n_high)
        self.rng.shuffle(quality)
        self.applicants = [Agent(i, q) for i, q in enumerate(quality)]
        self.employers = [Agent(i, "Employer") for i in range(self.p.employers)]
        self.groups = {
            "HQ": [a for a in self.applicants if a.quality == "HQ"],
            "LQ": [a for a in self.applicants if a.quality == "LQ"],
            "Employer": self.employers,
        }
        for group, center in zip(self.groups.values(),
                                 (self.p.initial_high, self.p.initial_low, self.p.initial_employer)):
            if self.p.initial_profile == "centered":
                half = self.p.initial_spread / 2
                lower, upper = max(0.0, center - half), min(1.0, center + half)
                for a in group:
                    a.technology = self.rng.uniform(lower, upper)
            else:
                count = int(math.floor(self.p.initial_majority_share * len(group) + 0.5))
                majority = set(self.rng.sample(group, count))
                bounds = (((0.8, 1.0), (0.0, 0.8)) if self.p.initial_profile == "high_mixture"
                          else ((0.0, 0.2), (0.2, 1.0)))
                for a in group:
                    a.technology = self.rng.uniform(*bounds[a not in majority])
        self.agents = self.applicants + self.employers

    def step(self) -> dict:
        if self.pending is not None:
            raise ValueError("Resolve all applicant decisions before advancing")
        if self.round >= self.rounds:
            raise ValueError("The market is complete")
        self.round += 1
        for a in self.agents:
            a.used = a.technology
            a.gross = a.cost = a.net = 0.0
            a.hired = False
        teams = [[] for _ in self.employers]
        available = list(self.applicants)
        for _ in range(self.p.positions):
            order = list(self.employers)
            self.rng.shuffle(order)
            for employer in order:
                weights = [hiring_weight(a.quality, a.used, employer.used, self.p)
                           for a in available]
                if available and sum(weights) > 0:
                    chosen = self.rng.choices(available, weights=weights, k=1)[0]
                    available.remove(chosen)
                    teams[employer.index].append(chosen)
                    chosen.hired = True
        output = undistributed = 0.0
        for employer, team in zip(self.employers, teams):
            production = sum(self.p.high_productivity if a.quality == "HQ"
                             else self.p.low_productivity for a in team)
            output += production
            employer.gross = 0.5 * production
            units = 1 if self.p.employer_cost_basis == "round" else len(team)
            employer.cost = self.p.employer_cost * units * (1.0 - employer.used)
            wage = production / (2 * self.p.positions)
            for a in team:
                a.gross = wage
            undistributed += 0.5 * production - len(team) * wage
        for a in self.applicants:
            pays_cost = self.p.applicant_cost_basis == "submission" or a.hired
            a.cost = self.p.applicant_cost * (1.0 - a.used) if pays_cost else 0.0
        for a in self.agents:
            a.net = a.gross - a.cost
            a.window += a.net
            a.cumulative += a.net
        result = {"round": self.round, "output": output,
                  "total_cost": sum(a.cost for a in self.agents),
                  "total_net": sum(a.net for a in self.agents),
                  "hired": sum(a.hired for a in self.applicants),
                  "undistributed_worker_share": undistributed}
        result["vacancies"] = self.p.employers * self.p.positions - result["hired"]
        result["accounting_residual"] = output - result["total_cost"] - result["total_net"] - undistributed
        result["high_share_hired"] = (sum(a.hired for a in self.groups["HQ"]) / result["hired"]
                                       if result["hired"] else math.nan)
        for label, group in self.groups.items():
            result[label] = sum(a.used for a in group) / len(group) if group else math.nan
            result[label + "_net"] = sum(a.net for a in group) / len(group) if group else math.nan
            if label != "Employer":
                result[label + "_hiring"] = sum(a.hired for a in group) / len(group) if group else math.nan
        if self.round % self.p.learning_interval == 0:
            self._learn()
        for label, group in self.groups.items():
            result[label + "_next"] = sum(a.technology for a in group) / len(group) if group else math.nan
        return result

    def _learn(self):
        observations = [None] * len(self.applicants)
        proposals = []
        payoff = "window" if self.p.payoff_memory == "window" else "cumulative"
        for group in self.groups.values():
            for a in group:
                peer = self.rng.choice([other for other in group if other is not a]) if len(group) > 1 else None
                strength = (self.p.employer_selection_strength if a.quality == "Employer"
                            else self.p.selection_strength)
                adopt = peer is not None and self.rng.random() < imitation_probability(
                    getattr(a, payoff), getattr(peer, payoff), strength)
                proposal = peer.technology if adopt else a.technology
                if a.quality != "Employer" and self.policy == "llm":
                    proposal = a.technology
                    own = {"technology_use": a.technology,
                           "last_rounds_net_payoff": a.window}
                    obs = {"completed_rounds": self.round,
                           "next_rounds": [self.round + 1, min(self.rounds, self.round + self.p.learning_interval)],
                           "own": own}
                    if self.information == "social" and peer is not None:
                        obs["other_applicant"] = {"technology_use": peer.technology,
                                                  "last_rounds_net_payoff": peer.window}
                    observations[a.index] = obs
                proposals.append((a, proposal))
        for a, proposal in proposals:
            a.technology = proposal
            a.window = 0.0
        if self.policy == "llm" and self.round < self.rounds:
            self.pending = observations

    def accept(self, values):
        if self.pending is None or len(values) != len(self.applicants):
            raise ValueError("A decision is required for every applicant at the barrier")
        if any(type(v) not in (float, int) or not math.isfinite(v) or not 0 <= v <= 1
               for v in values):
            raise ValueError("Every action must be a finite number in [0, 1]")
        for a, value in zip(self.applicants, values):
            a.technology = float(value)
        self.pending = None
