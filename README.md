# graph-theoretic-networking
This project models professional networking as a graph optimization problem. The goal is to maximize two-hop reach under a limited connection budget.

Key ideas:
- Two-hop reach as an objective function
- Maximum coverage formulation
- Comparison of degree-based vs greedy strategies
- Simulations on scale-free and community-structured graphs

This began as a personal observation during a job search and evolved into an applied network science case study.

PDF write-up included.

## Potential Limitations & Extensions

**Partial Observability** – The paper acknowledges but doesn’t deeply simulate the challenge that users only see part of the network (e.g., only 1st-degree connections’ profiles). This could be expanded.

**Dynamic Networks** – Professional networks evolve over time; a longitudinal strategy could be interesting.

**Weighted Edges** – Not all connections are equal. Incorporating tie strength (frequency of interaction, trust) could refine the model.

**Alternative Objectives** – Besides two-hop reach, one might optimize for influence spread, job referral likelihood, or diversity of access.

**Real-Data Validation** – While synthetic simulations are clean, testing on real LinkedIn (anonymized) data would strengthen conclusions.
