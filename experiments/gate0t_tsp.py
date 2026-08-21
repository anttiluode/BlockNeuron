from __future__ import annotations

"""Gate 0T — can hysteretic BlockNeuron-style matter solve a tiny TSP?

This is a deliberately small proof-of-principle, not a competitive TSP solver.
It maps a Euclidean travelling-salesman problem onto an n x n continuous
city-by-tour-position field. A Sinkhorn readout turns the field into a soft
permutation. The hysteretic solver evolves a polarization-like state p using

    dp/dt = p - p^3 + E_tsp
    E_tsp = -gain * d(route_cost)/dp

so the ordinary TSP objective acts as a field on double-well local states.
After the dynamics stop, the soft assignment is projected to a valid tour.

Controls:
  * plain continuous relaxation: same Sinkhorn/TSP objective, no double well;
  * multi-start nearest-neighbour + 2-opt heuristic;
  * exact exhaustive optimum for these tiny instances.

The important stop line is simple: solving a small TSP does not establish a
BlockNeuron advantage. This construction is close in spirit to classical
Hopfield/analog optimization. If the ordinary relaxer or 2-opt matches or beats
it, the result is capability only.
"""

import argparse
import itertools
import math
import statistics
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class SolverReceipt:
    length: float
    order: tuple[int, ...]
    polarization_abs_mean: float = float("nan")
    assignment_peak_mean: float = float("nan")


def euclidean_instance(n: int, seed: int) -> tuple[Tensor, Tensor]:
    g = torch.Generator().manual_seed(int(seed))
    points = torch.rand(n, 2, generator=g)
    distance = torch.cdist(points, points)
    return points, distance


def tour_length(order: tuple[int, ...] | list[int], distance: Tensor) -> float:
    n = len(order)
    return sum(float(distance[order[k], order[(k + 1) % n]]) for k in range(n))


def exact_tsp(distance: Tensor) -> SolverReceipt:
    """Exact optimum with city 0 fixed as the first city to remove rotations."""
    n = int(distance.shape[0])
    if n > 9:
        raise ValueError("exact exhaustive receipt is intentionally limited to n<=9")
    best_length = math.inf
    best_order: tuple[int, ...] | None = None
    for suffix in itertools.permutations(range(1, n)):
        order = (0,) + suffix
        length = tour_length(order, distance)
        if length < best_length:
            best_length = length
            best_order = order
    assert best_order is not None
    return SolverReceipt(best_length, best_order)


def sinkhorn(logits: Tensor, iterations: int = 8) -> Tensor:
    """Exponentiated log-space Sinkhorn normalization."""
    z = logits
    for _ in range(int(iterations)):
        z = z - torch.logsumexp(z, dim=1, keepdim=True)
        z = z - torch.logsumexp(z, dim=0, keepdim=True)
    return z.exp()


def soft_tour_cost(assignment: Tensor, distance: Tensor) -> Tensor:
    """Expected cyclic route cost of a soft city x position assignment."""
    n = int(assignment.shape[1])
    total = assignment.new_zeros(())
    for pos in range(n):
        a = assignment[:, pos][:, None]
        b = assignment[:, (pos + 1) % n][None, :]
        total = total + (a * b * distance).sum()
    return total


def project_assignment(assignment: Tensor) -> tuple[int, ...]:
    """Project a tiny soft assignment to the maximum-score valid permutation."""
    n = int(assignment.shape[0])
    if n > 8:
        raise ValueError("exact assignment projection is intentionally limited to n<=8")
    a = assignment.detach().cpu()
    best_score = -math.inf
    best_order: tuple[int, ...] | None = None
    for order in itertools.permutations(range(n)):
        score = sum(float(a[order[pos], pos]) for pos in range(n))
        if score > best_score:
            best_score = score
            best_order = order
    assert best_order is not None
    return best_order


def material_tsp(
    distance: Tensor,
    *,
    seed: int,
    restarts: int = 8,
    steps: int = 300,
    dt: float = 0.05,
    field_gain: float = 5.0,
    beta_start: float = 0.5,
    beta_end: float = 6.0,
) -> SolverReceipt:
    """Landau/double-well relaxation under a differentiable TSP field."""
    n = int(distance.shape[0])
    best: SolverReceipt | None = None

    for restart in range(int(restarts)):
        g = torch.Generator().manual_seed(int(seed) * 1009 + restart * 17 + 71)
        p = 0.05 * torch.randn(n, n, generator=g)

        for step in range(int(steps)):
            p = p.detach().requires_grad_(True)
            frac = step / max(steps - 1, 1)
            beta = beta_start + (beta_end - beta_start) * frac
            assignment = sinkhorn(beta * p)
            cost = soft_tour_cost(assignment, distance)
            grad = torch.autograd.grad(cost, p)[0]
            with torch.no_grad():
                # Landau gradient flow plus a task-derived external field.
                p = p + dt * (p - p.pow(3) - field_gain * grad)
                p.clamp_(-1.6, 1.6)

        with torch.no_grad():
            assignment = sinkhorn(beta_end * p, iterations=20)
            order = project_assignment(assignment)
            length = tour_length(order, distance)
            receipt = SolverReceipt(
                length=length,
                order=order,
                polarization_abs_mean=float(p.abs().mean()),
                assignment_peak_mean=float(assignment.max(dim=0).values.mean()),
            )
            if best is None or receipt.length < best.length:
                best = receipt

    assert best is not None
    return best


def plain_relaxation_tsp(
    distance: Tensor,
    *,
    seed: int,
    restarts: int = 8,
    steps: int = 300,
    learning_rate: float = 1.5,
    beta_start: float = 0.5,
    beta_end: float = 6.0,
) -> SolverReceipt:
    """Matched soft-permutation relaxation without hysteretic double wells."""
    n = int(distance.shape[0])
    best: SolverReceipt | None = None

    for restart in range(int(restarts)):
        g = torch.Generator().manual_seed(int(seed) * 1013 + restart * 19 + 1007)
        q = 0.05 * torch.randn(n, n, generator=g)

        for step in range(int(steps)):
            q = q.detach().requires_grad_(True)
            frac = step / max(steps - 1, 1)
            beta = beta_start + (beta_end - beta_start) * frac
            assignment = sinkhorn(beta * q)
            cost = soft_tour_cost(assignment, distance)
            grad = torch.autograd.grad(cost, q)[0]
            with torch.no_grad():
                q = q - learning_rate * grad
                q.clamp_(-5.0, 5.0)

        with torch.no_grad():
            assignment = sinkhorn(beta_end * q, iterations=20)
            order = project_assignment(assignment)
            length = tour_length(order, distance)
            receipt = SolverReceipt(
                length=length,
                order=order,
                polarization_abs_mean=float(q.abs().mean()),
                assignment_peak_mean=float(assignment.max(dim=0).values.mean()),
            )
            if best is None or receipt.length < best.length:
                best = receipt

    assert best is not None
    return best


def nearest_neighbour(distance: Tensor, start: int) -> tuple[int, ...]:
    n = int(distance.shape[0])
    remaining = set(range(n))
    remaining.remove(int(start))
    order = [int(start)]
    current = int(start)
    while remaining:
        nxt = min(remaining, key=lambda j: float(distance[current, j]))
        order.append(nxt)
        remaining.remove(nxt)
        current = nxt
    return tuple(order)


def two_opt(order: tuple[int, ...], distance: Tensor) -> SolverReceipt:
    current = list(order)
    n = len(current)
    best_length = tour_length(current, distance)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for k in range(i + 1, n):
                candidate = current[:i] + list(reversed(current[i : k + 1])) + current[k + 1 :]
                length = tour_length(candidate, distance)
                if length + 1e-12 < best_length:
                    current = candidate
                    best_length = length
                    improved = True
                    break
            if improved:
                break
    return SolverReceipt(best_length, tuple(current))


def heuristic_tsp(distance: Tensor) -> SolverReceipt:
    best: SolverReceipt | None = None
    for start in range(int(distance.shape[0])):
        receipt = two_opt(nearest_neighbour(distance, start), distance)
        if best is None or receipt.length < best.length:
            best = receipt
    assert best is not None
    return best


def mean_sd(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def run_suite(
    *,
    cities: int,
    instances: int,
    restarts: int,
    steps: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    if cities > 8:
        raise ValueError("this toy executable uses exact assignment projection and is limited to <=8 cities")

    ratios: dict[str, list[float]] = {"material": [], "plain": [], "two_opt": []}
    exact_hits = {"material": 0, "plain": 0, "two_opt": 0}
    material_peaks: list[float] = []
    material_abs: list[float] = []

    print("Gate 0T — hysteretic matter on tiny Euclidean TSP")
    print(f"cities={cities} instances={instances} restarts={restarts} steps={steps}")
    print("projection makes every reported candidate a valid Hamiltonian tour")
    print()
    print("instance   optimum    material   plain-relax   NN+2opt    material/opt")

    for index in range(instances):
        instance_seed = seed + index * 97
        _, distance = euclidean_instance(cities, instance_seed)
        optimum = exact_tsp(distance)
        material = material_tsp(
            distance,
            seed=instance_seed,
            restarts=restarts,
            steps=steps,
        )
        plain = plain_relaxation_tsp(
            distance,
            seed=instance_seed,
            restarts=restarts,
            steps=steps,
        )
        heuristic = heuristic_tsp(distance)

        for name, receipt in [
            ("material", material),
            ("plain", plain),
            ("two_opt", heuristic),
        ]:
            ratio = receipt.length / optimum.length
            ratios[name].append(ratio)
            exact_hits[name] += int(abs(receipt.length - optimum.length) <= 1e-6)

        material_peaks.append(material.assignment_peak_mean)
        material_abs.append(material.polarization_abs_mean)
        print(
            f"{index:8d}  {optimum.length:8.4f}  {material.length:8.4f}  "
            f"{plain.length:11.4f}  {heuristic.length:8.4f}      "
            f"{material.length/optimum.length:7.4f}"
        )

    summary: dict[str, dict[str, float]] = {}
    print()
    print("summary (tour length / exact optimum; lower is better, 1.0 is exact)")
    print("solver        mean ratio ± sd      exact hit rate")
    for name in ["material", "plain", "two_opt"]:
        mean, sd = mean_sd(ratios[name])
        hit_rate = exact_hits[name] / instances
        summary[name] = {
            "mean_ratio": mean,
            "sd_ratio": sd,
            "exact_hit_rate": hit_rate,
        }
        print(f"{name:10s}   {mean:7.4f} ± {sd:7.4f}        {hit_rate:7.4f}")

    summary["material"]["assignment_peak_mean"] = statistics.mean(material_peaks)
    summary["material"]["polarization_abs_mean"] = statistics.mean(material_abs)
    print()
    print(
        "material state diagnostics: "
        f"mean |p|={statistics.mean(material_abs):.4f}  "
        f"mean assignment peak={statistics.mean(material_peaks):.4f}"
    )

    capable = (
        summary["material"]["mean_ratio"] <= 1.05
        and summary["material"]["exact_hit_rate"] >= 0.50
    )
    print()
    print(f"toy TSP capability verdict: {'PASS' if capable else 'FAIL'}")
    if summary["material"]["mean_ratio"] >= min(
        summary["plain"]["mean_ratio"], summary["two_opt"]["mean_ratio"]
    ) - 1e-9:
        print("advantage verdict: NO — an ordinary control ties or beats the material solver")
    else:
        print("advantage verdict: UNRESOLVED — material is better on this tiny receipt; attack harder")
    print(
        "claim boundary: this only asks whether continuous hysteretic state can host a tiny "
        "TSP relaxation. It does not establish scaling, novelty, or competitiveness."
    )

    if not capable:
        raise SystemExit(1)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cities", type=int, default=7)
    parser.add_argument("--instances", type=int, default=6)
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=23_001)
    args = parser.parse_args()
    run_suite(
        cities=args.cities,
        instances=args.instances,
        restarts=args.restarts,
        steps=args.steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
