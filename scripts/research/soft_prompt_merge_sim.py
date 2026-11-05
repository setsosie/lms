#!/usr/bin/env python3
"""Merge operators on soft prompts, on synthetic embeddings (no model, no Lean).

Part A, neutral drift: no selection. Do merges keep soft tokens real-token sized
and keep the population spread out? Ratios are against a real token, as in the
soft-prompt spec's collapse probe.

Parts B and C, toy search: a hidden target direction with noisy fitness and a
budget of 500 evaluations. The landscape is smooth, so these show operator
mechanics and the effect of genome size, not what Lean fitness will do.

Run: uv run --no-project --with numpy python scripts/research/soft_prompt_merge_sim.py
"""

import numpy as np

V, D, K, POP, GENS = 50_000, 256, 8, 24, 20
SIGMA = 0.1  # per-coordinate mutation, against unit-variance token coordinates


def make_vocab(rng):
    embeddings = rng.standard_normal((V, D)).astype(np.float32)
    centroid = embeddings.mean(axis=0)
    radius = float(np.linalg.norm(embeddings - centroid, axis=1).mean())
    return embeddings, centroid, radius


def k_hot(rng, embeddings, n, active=2):
    return embeddings[rng.integers(0, V, size=(n, K, active))].mean(axis=2)


def merge_mean(rng, pop, parents, centroid, keep_norm=False):
    chosen = pop[rng.integers(0, len(pop), size=(len(pop), parents))]
    child = chosen.mean(axis=1)
    if keep_norm:
        target = np.linalg.norm(chosen - centroid, axis=-1).mean(axis=1)[..., None]
        now = np.linalg.norm(child - centroid, axis=-1, keepdims=True)
        child = centroid + (child - centroid) * target / np.maximum(now, 1e-6)
    return child


def merge_per_token(rng, pop, parents):
    n = len(pop)
    chosen = rng.integers(0, n, size=(n, parents))
    pick = rng.integers(0, parents, size=(n, K))
    return pop[chosen[np.arange(n)[:, None], pick], np.arange(K)[None, :]]


def spread_stats(pop, centroid, radius):
    size = np.linalg.norm(pop - centroid, axis=-1).mean() / radius
    i, j = np.triu_indices(len(pop), k=1)
    spread = np.linalg.norm(pop[i] - pop[j], axis=-1).mean()
    return size, spread / (np.sqrt(2) * radius)


OPERATORS = [
    ("mutation only", None, 0, False),
    ("mean, 2 parents", "mean", 2, False),
    ("mean, 3 parents", "mean", 3, False),
    ("mean, 5 parents", "mean", 5, False),
    ("mean, 2, norm kept", "mean", 2, True),
    ("mean, 5, norm kept", "mean", 5, True),
    ("per-token, 2 parents", "token", 2, False),
    ("per-token, 5 parents", "token", 5, False),
]


def apply_operator(rng, pop, centroid, kind, parents, keep_norm):
    if kind == "mean":
        return merge_mean(rng, pop, parents, centroid, keep_norm)
    if kind == "token":
        return merge_per_token(rng, pop, parents)
    return pop


def drift(start, kind, parents, keep_norm, mutate):
    rng = np.random.default_rng(0)
    embeddings, centroid, radius = make_vocab(rng)
    if start == "independent":
        pop = k_hot(rng, embeddings, POP)
    else:
        pop = np.repeat(k_hot(rng, embeddings, 1), POP, axis=0)
        pop = pop + SIGMA * rng.standard_normal(pop.shape)
    first = spread_stats(pop, centroid, radius)
    for _ in range(GENS):
        pop = apply_operator(rng, pop, centroid, kind, parents, keep_norm)
        if mutate:
            pop = pop + SIGMA * rng.standard_normal(pop.shape)
    return first, spread_stats(pop, centroid, radius)


def part_a(start):
    print(
        f"\nPart A, start = {start}: (size, spread) against a real token, sigma={SIGMA}"
    )
    print(f"{'operator':<24}{'gen 0':>14}{'gen 20':>14}{'gen 20, no mutation':>22}")
    for name, kind, parents, keep_norm in OPERATORS:
        first, with_mut = drift(start, kind, parents, keep_norm, True)
        _, without = drift(start, kind, parents, keep_norm, False)
        print(
            f"{name:<24}{first[0]:>8.2f},{first[1]:>5.2f}"
            f"{with_mut[0]:>8.2f},{with_mut[1]:>5.2f}{without[0]:>16.2f},{without[1]:>5.2f}"
        )


def search(dim, noise, kind, parents, weighted, seed, evals=500, offspring=10):
    rng = np.random.default_rng(seed)
    target = rng.standard_normal(dim)
    target /= np.linalg.norm(target)
    start = rng.standard_normal((parents, dim))
    pop = start / np.linalg.norm(start, axis=1, keepdims=True)
    weights = np.ones(parents) / parents
    if weighted:
        weights = np.log(parents + 1) - np.log(np.arange(1, parents + 1))
        weights /= weights.sum()
    for _ in range(evals // offspring):
        if kind == "mean":
            bases = np.repeat((weights @ pop)[None, :], offspring, axis=0)
        elif kind == "crossover":
            first = pop[rng.integers(0, parents, size=offspring)]
            second = pop[rng.integers(0, parents, size=offspring)]
            bases = np.where(rng.random((offspring, dim)) < 0.5, first, second)
        else:
            bases = pop[rng.integers(0, parents, size=offspring)]
        kids = bases + 0.5 * rng.standard_normal((offspring, dim)) / np.sqrt(dim)
        kids /= np.linalg.norm(kids, axis=1, keepdims=True)
        fitness = kids @ target + noise * rng.standard_normal(offspring)
        pop = kids[np.argsort(fitness)[::-1][:parents]]
    final = weights @ pop
    return float(final / np.linalg.norm(final) @ target)


def table(title, columns):
    print(f"\n{title}")
    print(f"{'genome size':>12}{'noise':>7}" + "".join(f"{c[0]:>12}" for c in columns))
    for dim in (16, 256, 2048, 32768):
        for noise in (0.0, 0.05):
            cells = [
                np.mean([search(dim, noise, c[1], c[2], c[3], s) for s in range(20)])
                for c in columns
            ]
            print(f"{dim:>12}{noise:>7}" + "".join(f"{x:>12.3f}" for x in cells))


if __name__ == "__main__":
    part_a("independent")
    part_a("one ancestor")
    table(
        "Part B, operators: final cosine with the target, 3 parents, 500 evals, 20 seeds",
        [
            ("mutation", "none", 3, False),
            ("crossover", "crossover", 3, False),
            ("mean", "mean", 3, False),
        ],
    )
    table(
        "Part C, parents in the mean: final cosine, 10 offspring, 500 evals, 20 seeds",
        [
            ("1", "mean", 1, False),
            ("2", "mean", 2, False),
            ("3", "mean", 3, False),
            ("5", "mean", 5, False),
            ("5 ranked", "mean", 5, True),
            ("10, no sel.", "mean", 10, False),
        ],
    )
