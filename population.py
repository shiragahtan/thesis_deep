"""
population.py
-------------
Manages the population of candidate programs.

A Program is just a (code, score, eval_result) triple.
The Population keeps them sorted by score and exposes
selection / replacement operations used by the main loop.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from benchmark import EvaluationResult


@dataclass
class Program:
    """One candidate in the population."""
    code:        str
    score:       float
    eval_result: EvaluationResult
    generation:  int = 0           # which generation produced this program

    def __lt__(self, other: "Program") -> bool:
        return self.score < other.score


class Population:
    """
    Fixed-size population of Programs, always sorted best-first.

    Key design choices:
      - Size is capped at POPULATION_SIZE; weakest program is evicted when full.
      - Parents are selected from the top-K best programs.
      - Exposes the full failure evidence of the best program for LLM prompting.
    """

    def __init__(self, max_size: int, top_k: int):
        self.max_size  = max_size
        self.top_k     = top_k
        self._programs: list[Program] = []

    # ── Mutating operations ────────────────────────────────────────────────────

    def add(self, program: Program) -> bool:
        """
        Add a program to the population.
        If the population is full, evict the worst program
        only if the new one is better.
        Returns True if the program was accepted.
        """
        if len(self._programs) < self.max_size:
            self._programs.append(program)
            self._sort()
            return True

        worst = self._programs[-1]
        if program.score > worst.score:
            self._programs[-1] = program
            self._sort()
            return True

        return False

    def _sort(self):
        self._programs.sort(key=lambda p: p.score, reverse=True)

    # ── Read operations ────────────────────────────────────────────────────────

    def best(self) -> Program:
        return self._programs[0]

    def top_parents(self) -> list[Program]:
        """Return the top-K programs used as parents for the next generation."""
        return self._programs[: self.top_k]

    def select_parent(self, rng) -> Program:
        """
        Sample one parent from the top-K, weighted by score.
        Higher-scoring parents are more likely to be selected.
        """
        parents = self.top_parents()
        scores  = [p.score for p in parents]
        total   = sum(scores) or 1.0
        weights = [s / total for s in scores]
        return rng.choices(parents, weights=weights, k=1)[0]

    def all_scores(self) -> list[float]:
        return [p.score for p in self._programs]

    def size(self) -> int:
        return len(self._programs)

    def diversity(self) -> float:
        """
        Simple diversity metric: std-dev of scores.
        Low diversity → consider injecting dumb steps.
        """
        scores = self.all_scores()
        if len(scores) < 2:
            return 0.0
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        return variance ** 0.5

    def __repr__(self) -> str:
        scores = [f"{p.score:.1f}" for p in self._programs]
        return f"Population(size={self.size()}, scores=[{', '.join(scores)}])"
