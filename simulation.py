import random
from collections import defaultdict

import matplotlib.pyplot as plt

try:
    import networkx as nx
except ImportError as e:
    raise ImportError(
        "This simulation requires networkx. Install with: pip install networkx"
    ) from e


def two_hop_reach(G, u):
    """
    Two-hop neighborhood size excluding u.
    N2(u) = union of neighbors of neighbors plus neighbors, excluding u.
    """
    nbrs = set(G.neighbors(u))
    two_hop = set(nbrs)
    for v in nbrs:
        two_hop.update(G.neighbors(v))
    two_hop.discard(u)
    return two_hop


def marginal_gain_twohop(G, u, v, current_twohop=None):
    """
    If u connects to v, the new nodes that enter u's two-hop set are roughly N(v) \ current_twohop.
    This is an approximation aligned with the maximum coverage framing.
    """
    if current_twohop is None:
        current_twohop = two_hop_reach(G, u)
    return len(set(G.neighbors(v)) - current_twohop)


def pick_candidates(G, u, max_candidates=500, seed=0):
    """
    Candidates are nodes not already connected to u and not u itself.
    To mimic LinkedIn search limits, we optionally downsample to max_candidates.
    """
    rng = random.Random(seed)
    forbidden = set(G.neighbors(u)) | {u}
    candidates = [v for v in G.nodes() if v not in forbidden]
    if len(candidates) > max_candidates:
        candidates = rng.sample(candidates, max_candidates)
    return candidates


def strategy_random(G, u, candidates, k, seed=0):
    rng = random.Random(seed)
    return rng.sample(candidates, k)


def strategy_high_degree(G, u, candidates, k):
    return sorted(candidates, key=lambda v: G.degree(v), reverse=True)[:k]


def strategy_high_betweenness(G, u, candidates, k, betweenness=None):
    if betweenness is None:
        # Exact betweenness can be expensive for large graphs.
        # For larger graphs, consider nx.betweenness_centrality(G, k=200, seed=0) for approximation.
        betweenness = nx.betweenness_centrality(G)
    return sorted(candidates, key=lambda v: betweenness[v], reverse=True)[:k]


def strategy_greedy_coverage(G, u, candidates, k):
    """
    Greedy maximum coverage on the sets N(v) over candidates:
    iteratively pick the v that adds the most new nodes to u's current two-hop set.
    """
    chosen = []
    current_twohop = two_hop_reach(G, u)

    remaining = set(candidates)
    for _ in range(k):
        best_v = None
        best_gain = -1

        for v in remaining:
            gain = marginal_gain_twohop(G, u, v, current_twohop=current_twohop)
            if gain > best_gain:
                best_gain = gain
                best_v = v

        if best_v is None:
            break

        chosen.append(best_v)
        # Update approximation of u's two-hop reach after adding (u, best_v)
        current_twohop |= set(G.neighbors(best_v))
        remaining.remove(best_v)

    return chosen


def apply_new_connections_and_measure(G, u, chosen):
    """
    Add edges (u, v) and measure two-hop reach size before and after.
    Operates on a copy to keep runs comparable.
    """
    H = G.copy()
    before = len(two_hop_reach(H, u))

    for v in chosen:
        H.add_edge(u, v)

    after = len(two_hop_reach(H, u))
    return before, after, after - before


def run_experiment(
    n=5000,
    m=3,
    k=10,
    trials=20,
    max_candidates=500,
    seed=0,
    compute_betweenness=False,
):
    """
    n: number of nodes
    m: BA parameter (new node attaches to m existing nodes)
    k: number of new connections you can add
    trials: number of different ego nodes sampled
    max_candidates: candidate pool cap
    compute_betweenness: if True, compute exact betweenness (slow for large n)
    """
    rng = random.Random(seed)

    # Scale-free network via preferential attachment
    G = nx.barabasi_albert_graph(n=n, m=m, seed=seed)

    # Optional betweenness cache
    bet = None
    if compute_betweenness:
        bet = nx.betweenness_centrality(G)

    results = defaultdict(list)

    nodes = list(G.nodes())
    for t in range(trials):
        u = rng.choice(nodes)

        candidates = pick_candidates(G, u, max_candidates=max_candidates, seed=seed + t)

        # Ensure we have enough candidates
        if len(candidates) < k:
            continue

        chosen_random = strategy_random(G, u, candidates, k, seed=seed + 1000 + t)
        chosen_degree = strategy_high_degree(G, u, candidates, k)
        chosen_greedy = strategy_greedy_coverage(G, u, candidates, k)

        if compute_betweenness:
            chosen_between = strategy_high_betweenness(G, u, candidates, k, betweenness=bet)
        else:
            chosen_between = None

        for name, chosen in [
            ("random", chosen_random),
            ("high_degree", chosen_degree),
            ("greedy_coverage", chosen_greedy),
        ]:
            before, after, gain = apply_new_connections_and_measure(G, u, chosen)
            results[name].append((before, after, gain))

        if chosen_between is not None:
            before, after, gain = apply_new_connections_and_measure(G, u, chosen_between)
            results["high_betweenness"].append((before, after, gain))

    return results


def summarize(results):
    summary = {}
    for name, rows in results.items():
        if not rows:
            continue
        befores = [r[0] for r in rows]
        afters = [r[1] for r in rows]
        gains = [r[2] for r in rows]
        summary[name] = {
            "runs": len(rows),
            "avg_before": sum(befores) / len(befores),
            "avg_after": sum(afters) / len(afters),
            "avg_gain": sum(gains) / len(gains),
        }
    return summary


def plot_gains(results):
    names = list(results.keys())
    avg_gains = []
    for name in names:
        rows = results[name]
        gains = [r[2] for r in rows]
        avg_gains.append(sum(gains) / len(gains) if gains else 0)

    plt.figure()
    plt.bar(names, avg_gains)
    plt.title("Average two-hop reach gain by strategy")
    plt.ylabel("Avg gain in |N2(u)|")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Recommended default run: fast, informative
    results = run_experiment(
        n=5000,          # network size
        m=3,             # attachment edges per new node
        k=10,            # you add 10 new connections
        trials=30,       # average over 30 different "you" nodes
        max_candidates=500,
        seed=42,
        compute_betweenness=False,  # True is slower
    )

    summary = summarize(results)
    for name, stats in sorted(summary.items(), key=lambda x: x[1]["avg_gain"], reverse=True):
        print(name, stats)

    plot_gains(results)
