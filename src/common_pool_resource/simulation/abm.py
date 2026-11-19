"""Resource harvesting, sanctions and payoff-based imitation."""

from dataclasses import dataclass, asdict
import random
import numpy as np

from ..config import Parameters


@dataclass
class Agent:
    index: int
    effort: float
    monitoring: float
    belief: float
    punishment_probability: float
    payoff: float = 0.0
    harvest: float = 0.0
    punished: bool = False
    was_punished: bool = False


class Society:
    def __init__(self, parameters=Parameters(), seed=0, rounds=101):
        if type(seed) is not int or not 0 <= seed < 2**32 or type(rounds) is not int or rounds < 1:
            raise ValueError("Use a 32-bit seed and a positive round count")
        self.p, self.seed, self.rounds = parameters, seed, rounds
        self.random, self.noise = random.Random(seed), np.random.RandomState(seed)
        self.resource, self.round, self.running = parameters.init_resource, 0, True
        self.agents = [Agent(i, self.random.random(), self.random.random() * 0.2,
                             self.random.uniform(2, 8), self.noise.uniform(parameters.punish_p_min,
                                                                         parameters.punish_p_min + 0.3))
                       for i in range(parameters.n_agents)]
        self.order = list(range(parameters.n_agents))

    def step(self):
        if not self.running:
            raise ValueError("Society has finished")
        p = self.p
        policy = float(np.median([a.belief for a in self.agents]))
        for a in self.agents:
            a.was_punished = False
            a.harvest = float(max(p.consumption, min(np.round(a.effort * self.resource * p.harvest_productivity), self.resource)))
            a.payoff = max(a.payoff + a.harvest - p.consumption, 0)
        self.random.shuffle(self.order)
        for i in self.order:
            a = self.agents[i]
            a.payoff -= a.monitoring * p.monitoring_cost
            a.punished = False
            peers = [j for j in self.order if j != i]
            other = self.agents[self.random.choice(peers)]
            if (other.harvest > policy and self.random.random() < a.punishment_probability
                    and not other.was_punished):
                other.was_punished = True
                other.payoff = max(other.payoff - p.penalty, 0)
                a.payoff -= p.punish_cost
                a.punished = True
            if (self.round + 1) % 3 == 0:
                other = self.agents[self.random.choice(peers)]
                if other.payoff > a.payoff:
                    for field, scale in (("effort", .1), ("monitoring", .1), ("belief", .1),
                                         ("punishment_probability", 1)):
                        sign = 1 if self.random.random() < .5 else -1
                        setattr(a, field, getattr(other, field) + sign * self.noise.normal(0, scale))
                a.effort = max(0., min(1., a.effort))
                a.monitoring = max(0., min(1., a.monitoring))
                a.belief = max(0., a.belief)
                a.punishment_probability = max(0., min(1., a.punishment_probability))
        demand = sum(a.harvest for a in self.agents)
        self.resource -= demand
        row = {"Step": self.round, "round": self.round + 1, "resource": self.resource,
               "total_demand": demand, "group_policy": policy}
        agents = [{"Step": self.round, **asdict(a)} for a in self.agents]
        if self.round > 0 and any(a.payoff < 0 for a in self.agents):
            self.running = False
        else:
            if self.resource <= 0:
                self.running = False
            self.resource = max(self.resource, 0)
            self.resource += p.growth_rate * self.resource
            self.resource = min(self.resource, p.capacity)
        self.round += 1
        self.running = self.running and self.round < self.rounds
        return row, agents
