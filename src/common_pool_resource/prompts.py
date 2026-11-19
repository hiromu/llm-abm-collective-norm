"""Fishing decisions and norm-revision instructions."""

EFFORT = """
You are a villager who fishes from a shared lake together with others in your community.
Each villager needs to consume {consumption} units of fish daily to survive. When your payoff becomes negative, you die.

Each villager holds a personal strategy about what they should do, and the community has also a shared policy.

Your personal strategy: "{agent_norm}"
Shared community policy: "{group_norm}"

You observe each villager's fishing effort and total payoff.
Based on both your personal belief and the community policy, decide how much effort you want to put into fishing today.

Respond with a single number between 0.0 and 1.0 in a new line. No explanation.
"""

PUNISHMENT = """
You are a villager who fishes from a shared lake together with others in your community.
Each villager needs to consume {consumption} units of fish daily to survive. When your payoff becomes negative, you die.

Each villager holds a personal strategy about what they should do, and the community has also a shared policy.

Your personal strategy: "{agent_norm}"
Shared community policy: "{group_norm}"

You observe each villager's fishing effort and total payoff.
Based on both your personal belief and the community policy, decide whether to punish others who violate these standards by deducting their payoff by {penalty}, while you need to pay {punish_cost} yourself.

Return only the ID number of the villager to punish, or "N/A" if no punishment is needed, in a new line. No explanation.
"""

NORM = """
You are a villager who fishes from a shared lake together with others in your community.
Each villager needs to consume {consumption} units of fish daily to survive. When your payoff becomes negative, you die.

Each villager holds a personal strategy about what they should do, and the community has also a shared policy.

Your personal strategy: "{agent_norm}"
Shared community policy: "{group_norm}"

You observe each villager's fishing effort and total payoff. Based on your observations:
1. Update your personal strategy about what you should do
2. Propose what the others should do in the community

Respond in exactly this format:
Personal: [Your updated personal belief]
Community: [Your proposed community policy]

No additional explanation.
"""

VOTE = """
You are a villager who fishes from a shared lake together with others in your community.
Each villager needs to consume {consumption} units of fish daily to survive. When your payoff becomes negative, you die.

Each villager holds a personal strategy about what they should do, and the community has also a shared policy.

Your personal strategy: "{agent_norm}"
Shared community policy: "{group_norm}"

Based on your personal strategy and the current state of the lake, vote for which proposed policy you think should become the new shared policy.

Respond with only the exact text of your chosen policy (copy it exactly as shown above). No explanation.
"""

INFORMATION = """
You:
{self}
Others:
{others}
Resource History: {resource_history} / Max Capacity: {capacity}
"""

ALTRUISTIC_NORMS = [
    'Fish only what you need to feed your family',
    'Preserve the lake for future generations',
    'Follow the community guidelines strictly',
    'Fish moderately to maintain the ecosystem',
    'Prioritize long-term sustainability over short-term gain',
    'Consider the needs of the entire community before fishing',
    'Leave enough fish for others and future generations',
    "Share the lake's resources fairly with all villagers",
]

SELFISH_NORMS = [
    'Maximize your catch while the fish are abundant',
    'Take as much as you can before others do',
    "Your family's needs come first, fish as much as possible",
    'The lake is there to be used, take your maximum share',
    'Fish aggressively to ensure your own survival',
    'Get the most value from your fishing effort',
    'Compete to harvest more than others',
    'Focus on your immediate gains from the lake',
]
