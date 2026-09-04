"""
scheduler.py
------------
Decides at each generation whether to use the Smart Step or Dumb Step.

Two strategies are implemented:

  FIXED_RATIO   — use smart step SMART_STEP_RATIO fraction of the time,
                  dumb step the rest. Simple baseline.

  DIVERSITY     — monitor population diversity. If diversity drops below a
                  threshold (diversity collapse), force a dumb step to
                  re-introduce variation. Otherwise use the smart step.

The scheduler is a key research variable:
  "What is the right balance of smart vs dumb steps,
   especially for hard tasks?" (Q3 from the thesis)
"""

from __future__ import annotations
from typing import Optional
import random
from config import SMART_STEP_RATIO


class Scheduler:
    """
    Decides smart vs dumb step for each generation.

    Args:
        strategy:          "fixed_ratio" | "diversity"
        smart_ratio:       fraction of smart steps (used by fixed_ratio)
        diversity_thresh:  if diversity < this, force a dumb step (used by diversity)
    """

    STRATEGIES = ("fixed_ratio", "diversity")

    def __init__(
        self,
        strategy: str = "fixed_ratio",
        smart_ratio: float = SMART_STEP_RATIO,
        diversity_thresh: float = 5.0,
        rng: Optional[random.Random] = None,
    ):
        assert strategy in self.STRATEGIES, f"Unknown strategy: {strategy}"
        self.strategy         = strategy
        self.smart_ratio      = smart_ratio
        self.diversity_thresh = diversity_thresh
        self.rng              = rng or random.Random()

        # tracking
        self.smart_count = 0
        self.dumb_count  = 0

    def use_smart_step(self, diversity: float | None = None) -> bool:
        """
        Return True  → use Smart Step (LLM)
        Return False → use Dumb Step (random)
        """
        if self.strategy == "fixed_ratio":
            decision = self.rng.random() < self.smart_ratio

        elif self.strategy == "diversity":
            if diversity is not None and diversity < self.diversity_thresh:
                # diversity too low → force a dumb step to shake things up
                decision = False
            else:
                decision = self.rng.random() < self.smart_ratio

        else:
            decision = True

        if decision:
            self.smart_count += 1
        else:
            self.dumb_count += 1

        return decision

    def summary(self) -> dict:
        total = self.smart_count + self.dumb_count or 1
        return {
            "strategy":    self.strategy,
            "smart_steps": self.smart_count,
            "dumb_steps":  self.dumb_count,
            "smart_pct":   round(100 * self.smart_count / total, 1),
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (f"Scheduler(strategy={s['strategy']}, "
                f"smart={s['smart_steps']}, dumb={s['dumb_steps']}, "
                f"smart%={s['smart_pct']})")
