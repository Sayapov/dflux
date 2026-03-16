#!/usr/bin/env python3
"""Δ_flux adapter for HuggingFace Transformers Trainer.

Usage:
    from transformers import Trainer
    from dflux.adapters.hf_trainer import DFluxCallback

    trainer = Trainer(
        ...,
        callbacks=[DFluxCallback(
            run_id="run-001",
            model_id="my-model-v1",
        )],
    )
    trainer.train()

Outputs:
    ./dflux_logs/dflux.csv
    ./dflux_logs/dflux_events.jsonl
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from transformers import TrainerCallback

from dflux.meter import DFluxMeter, DFluxConfig


def _auto_L_cut(n_groups: int) -> int:
    """Deep tail = last ~40% of parameter groups."""
    return max(0, int(0.6 * n_groups) - 1)


class DFluxCallback(TrainerCallback):
    """HuggingFace Trainer callback that runs the Δ_flux meter.

    Hooks into on_pre_optimizer_step (after backward/clipping,
    before optimizer.step) — the correct moment to read gradients.
    """

    def __init__(
        self,
        run_id: str = "run-unknown",
        model_id: str = "model-unknown",
        L_cut: Union[int, str] = "auto",
        window_steps: int = 200,
        theta_warning: float = 0.5,
        theta_critical: Optional[float] = None,
        log_every: int = 1,
        positive_only: bool = True,
        out_dir: str = "./dflux_logs",
    ):
        self.run_id = run_id
        self.model_id = model_id
        self.L_cut = L_cut
        self.window_steps = window_steps
        self.theta_warning = theta_warning
        self.theta_critical = theta_critical
        self.log_every = log_every
        self.positive_only = positive_only
        self.out_dir = out_dir
        self.meter: Optional[DFluxMeter] = None

    def on_train_begin(self, args, state, control, **kwargs):
        optimizer = kwargs.get("optimizer")
        if optimizer is None:
            return

        out = Path(self.out_dir)
        out.mkdir(parents=True, exist_ok=True)

        n_groups = len(optimizer.param_groups)
        L_cut = _auto_L_cut(n_groups) if self.L_cut == "auto" else int(self.L_cut)

        self.meter = DFluxMeter.from_optimizer(
            optimizer,
            L_cut=L_cut,
            window_steps=self.window_steps,
            theta_warning=self.theta_warning,
            theta_critical=self.theta_critical,
            positive_only=self.positive_only,
            log_every=self.log_every,
            csv_path=str(out / "dflux.csv"),
            events_path=str(out / "dflux_events.jsonl"),
            run_id=self.run_id,
            model_id=self.model_id,
            framework="pytorch",
        )

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if self.meter is None:
            return

        step = int(getattr(state, "global_step", 0))

        lr = None
        try:
            optimizer = kwargs.get("optimizer")
            if optimizer:
                lr = float(optimizer.param_groups[0].get("lr", 0))
        except Exception:
            lr = None

        loss = kwargs.get("loss")
        if loss is not None:
            try:
                loss = float(loss)
            except Exception:
                loss = None

        self.meter.step(step=step, loss=loss, lr=lr)

    def on_train_end(self, args, state, control, **kwargs):
        if self.meter is not None:
            self.meter.close()
            self.meter = None
