#!/usr/bin/env python3
"""Δ_flux adapter for PyTorch Lightning.

Usage:
    import lightning as L
    from dflux.adapters.lightning import DFluxCallback

    trainer = L.Trainer(
        callbacks=[DFluxCallback(
            run_id="run-001",
            model_id="my-model-v1",
        )],
    )
    trainer.fit(model, datamodule)

Outputs:
    ./dflux_logs/dflux.csv
    ./dflux_logs/dflux_events.jsonl
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional, Union

from dflux.meter import DFluxMeter, DFluxConfig

try:
    import lightning.pytorch as pl
except ImportError:
    try:
        import pytorch_lightning as pl
    except ImportError:
        raise ImportError(
            "PyTorch Lightning is required for DFluxCallback. "
            "Install with: pip install lightning"
        )


def _auto_L_cut(n_groups: int) -> int:
    """Deep tail = last ~40% of parameter groups."""
    return max(0, int(0.6 * n_groups) - 1)


class DFluxCallback(pl.Callback):
    """PyTorch Lightning callback that runs the Δ_flux meter.

    Hooks into on_before_optimizer_step — after backward,
    before optimizer.step. This is where gradients are readable.
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
        super().__init__()
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

    def on_fit_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        # Lightning may have multiple optimizers; we watch the first one.
        optimizers = trainer.optimizers
        if not optimizers:
            return

        optimizer = optimizers[0]
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

    def on_before_optimizer_step(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        optimizer: Any,
        *args,
        **kwargs,
    ) -> None:
        if self.meter is None:
            return

        step = trainer.global_step

        lr = None
        try:
            lr = float(optimizer.param_groups[0].get("lr", 0))
        except Exception:
            lr = None

        # Lightning tracks loss on the trainer
        loss = None
        try:
            loss = float(trainer.callback_metrics.get("train_loss", None))
        except Exception:
            loss = None

        self.meter.step(step=step, loss=loss, lr=lr)

    def on_fit_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if self.meter is not None:
            self.meter.close()
            self.meter = None
