"""State transitions for harvesting, sanctions, norm revision and voting."""

from collections import Counter
from dataclasses import asdict, dataclass, field
import math
import random
import re

from ..config import ABLATIONS, Parameters
from .. import prompts


@dataclass
class Agent:
    index: int
    effort: float
    individual_norm: str
    is_altruistic: bool
    harvest: float = 0.0
    payoff: float = 0.0
    dead: bool = False
    effort_history: list = field(default_factory=list)
    payoff_history: list = field(default_factory=list)
    punished_agent_id: int | None = None
    was_punished_by: list = field(default_factory=list)


class Society:
    def __init__(self, parameters=Parameters(), seed=0, rounds=51, condition="all"):
        if condition not in ABLATIONS or type(seed) is not int or not 0 <= seed < 2**32 or type(rounds) is not int or rounds < 1:
            raise ValueError("Invalid condition, seed or round count")
        self.p, self.seed, self.rounds, self.condition = parameters, seed, rounds, condition
        self.random = random.Random(seed)
        self.agents = []
        for i in range(parameters.n_agents):
            effort = self.random.random()
            altruistic = i < parameters.n_agents * parameters.altruism_ratio
            norm = self.random.choice(prompts.ALTRUISTIC_NORMS if altruistic else prompts.SELFISH_NORMS)
            self.agents.append(Agent(i, effort, norm, altruistic))
        self.order = list(range(parameters.n_agents))
        self.resource, self.resource_history = parameters.init_resource, []
        self.group_norm, self.proposals, self.votes = "N/A", [], []
        self.round, self.phase, self.turn = 0, "start", 0
        self.turn_order, self.pending, self.issues = [], [], []
        self.running = True

    def individual_payload(self, a, own=False):
        if a.dead:
            return f"- ID: {a.index}, Dead"
        if not own and "social learning" in ABLATIONS[self.condition]:
            return f"- ID: {a.index}, Alive"
        fields = [f"- ID: {a.index}"]
        for label, values in (("Effort", a.effort_history), ("Payoff", a.payoff_history)):
            if values:
                fields.append(f"{label}: {str(values).replace(']', ' (now)]')}")
        return ", ".join(fields)

    def messages(self, phase, index):
        a = self.agents[index]
        template = {"effort": prompts.EFFORT, "punish": prompts.PUNISHMENT,
                    "norm": prompts.NORM, "vote": prompts.VOTE}[phase]
        system = template.format(agent_norm=a.individual_norm, group_norm=self.group_norm,
                                 consumption=self.p.consumption, penalty=self.p.penalty,
                                 punish_cost=self.p.punish_cost)
        if "social learning" in ABLATIONS[self.condition]:
            system = re.sub(r"You observe each villager's fishing effort and total payoff.[ \n]*", "", system)
        if "group norm" in ABLATIONS[self.condition]:
            system = re.sub(r'Shared community policy: ".*"\n', "", system)
        user = prompts.INFORMATION.format(
            self=self.individual_payload(a, True),
            others="\n".join(self.individual_payload(self.agents[i]) for i in self.order if i != index),
            resource_history=str(self.resource_history).replace(']', ' (now)]'), capacity=self.p.capacity).strip()
        if phase == "vote":
            policies = self.random.sample(self.proposals, len(self.proposals))
            user += "\nProposed group norms:\n" + "\n".join(f"- {p['content']}" for p in policies)
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def requests(self):
        if not self.running:
            return []
        if self.pending:
            return self.pending
        if self.phase == "start":
            self.resource_history = (self.resource_history + [round(self.resource, 2)])[-5:]
            self.issues = []
            self.random.shuffle(self.order)
            for a in self.agents:
                a.was_punished_by = []
            self.phase = "effort"
        ids = (self.order if self.phase in ("effort", "vote") else [self.turn_order[self.turn]])
        self.pending = [{"key": f"{self.round}:{self.phase}:{i}", "agent": i,
                         "phase": self.phase, "messages": self.messages(self.phase, i)}
                        for i in ids if not self.agents[i].dead]
        return self.pending

    def accept(self, responses):
        """Apply one complete stage; responses retain their completion order."""
        if not self.pending or set(responses) != {r["key"] for r in self.pending}:
            raise ValueError("Every pending request needs exactly one response")
        by_key = {r["key"]: r for r in self.pending}
        for key, text in responses.items():
            a = self.agents[by_key[key]["agent"]]
            if self.phase == "effort":
                try:
                    value = float(text.strip(' \n"').split("\n")[-1])
                    if not math.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError
                    a.effort = value
                except ValueError:
                    self.issues.append({"agent": a.index, "phase": "effort", "fallback": "previous effort"})
            elif self.phase == "punish":
                a.punished_agent_id = None
                text = text.strip(' \n"').split("\n")[-1]
                match = re.match(r"[- ]*(ID:? *)?(?P<id>[0-9]+)", text)
                target = int(match.group("id")) if match else -1
                if target != a.index and 0 <= target < len(self.agents):
                    other = self.agents[target]
                    other.payoff = max(other.payoff - self.p.penalty, 0)
                    a.payoff -= self.p.punish_cost
                    a.punished_agent_id = target
                    other.was_punished_by.append(a.index)
                elif text.upper() != "N/A":
                    self.issues.append({"agent": a.index, "phase": "punish", "fallback": "no sanction"})
            elif self.phase == "norm":
                clean = re.sub(r"<think>[\s\S]*?</think>[\s]*?", "", text).strip()
                values = {}
                for line in clean.split("\n"):
                    for label in ("Personal", "Community"):
                        if line.startswith(label + ":"):
                            values[label] = line.replace(label + ":", "").strip(" \t-'\"[]")
                if len(values.get("Personal", "")) > 10:
                    a.individual_norm = values["Personal"]
                else:
                    self.issues.append({"agent": a.index, "phase": "norm", "fallback": "previous personal norm"})
                if len(values.get("Community", "")) > 10:
                    self.proposals.append({"id": a.index, "content": values["Community"]})
            else:
                self.votes.append(re.sub(r"<think>[\s\S]*?</think>[\s]*?", "", text).strip())
        self.pending = []
        if self.phase == "effort":
            for a in self.agents:
                if a.dead:
                    continue
                a.harvest = a.effort * self.resource * self.p.harvest_productivity
                a.payoff += a.harvest
                a.payoff -= self.p.consumption
                if a.payoff < 0:
                    a.dead, a.effort, a.harvest = True, 0, 0
                a.effort_history = (a.effort_history + [round(a.effort, 2)])[-5:]
                a.payoff_history = (a.payoff_history + [round(a.payoff, 2)])[-5:]
            alive = [i for i in self.order if not self.agents[i].dead]
            self.total_demand = sum(self.agents[i].harvest for i in alive)
            self.resource -= self.total_demand
            if alive and self.resource >= 0:
                self.proposals, self.votes = [], []
                self.turn_order, self.turn = self.random.sample(alive, len(alive)), 0
                self.phase = "punish"
                return None
            return self.finish()
        if self.phase == "punish":
            self.phase = "norm"
            return None
        if self.phase == "norm":
            self.turn += 1
            if self.turn < len(self.turn_order):
                self.phase = "punish"
                return None
            if self.proposals and "group norm" not in ABLATIONS[self.condition]:
                self.phase = "vote"
                return None
        elif self.votes:
            self.group_norm = Counter(self.votes).most_common(1)[0][0]
        return self.finish()

    def finish(self):
        row = {"Step": self.round, "round": self.round + 1, "resource": self.resource,
               "total_demand": self.total_demand, "group_norm": self.group_norm,
               "proposed_norms": self.proposals.copy(), "issues": self.issues.copy()}
        agents = [{"Step": self.round, **asdict(a)} for a in self.agents]
        self.running = not any(a.dead for a in self.agents) and self.resource >= 0
        if self.running:
            self.resource += self.p.growth_rate * self.resource * (1 - self.resource / self.p.capacity)
            self.resource = min(self.resource, self.p.capacity)
        self.round += 1
        self.running = self.running and self.round < self.rounds
        self.phase = "start"
        return row, agents

    def state(self):
        result = {k: v for k, v in vars(self).items() if k not in ("p", "random", "agents")}
        return {**result, "parameters": asdict(self.p), "random_state": self.random.getstate(),
                "agents": [asdict(a) for a in self.agents]}

    @classmethod
    def restore(cls, state):
        obj = cls.__new__(cls)
        obj.__dict__.update({k: v for k, v in state.items() if k not in ("parameters", "random_state", "agents")})
        obj.p = Parameters(**state["parameters"])
        obj.agents = [Agent(**a) for a in state["agents"]]
        obj.random = random.Random()
        version, values, gaussian = state["random_state"]
        obj.random.setstate((version, tuple(values), gaussian))
        return obj
