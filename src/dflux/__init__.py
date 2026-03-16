"""Δ_flux — Training stability meter, AXE-NS engine & inference probe.

Δ_flux: A general-purpose instability detector that treats gradient flow
as fluid dynamics. Watches for energy migration into deep parameter
bands and fires when sustained flux crosses a threshold.

AXE-NS: The full engine — detects, diagnoses, prescribes fixes, and
adaptively refines. Five-tower architecture (Sensitivity, Equivalence,
Memory, Agency, Identity) built on Navier-Stokes fluid analogues.

Inference Probe: Activation-space monitor for transformer inference.
Same fluid dynamics math, applied to hidden states instead of gradients.
Per-token regime classification and hallucination risk scoring.

Free instrument. MIT license.
"""

__version__ = "0.4.0"

from dflux.meter import DFluxMeter, DFluxConfig
from dflux.axe_ns import (
    AXEEngine,
    AXEConfig,
    Regime,
    Action,
    Witness,
    JState,
    classify_regime,
)
from dflux.inference_probe import (
    InferenceProbe,
    ProbeConfig,
    TokenDiagnostic,
)
from dflux.fine_probe import (
    FineProbe,
    FineProbeConfig,
    FineTokenDiagnostic,
)
from dflux.head_surgery import (
    HeadSurgeon,
    HeadIntervention,
    SurgeryReport,
)
from dflux.causal_primitives import (
    CausalPrimitives,
    CPConfig,
    HeadCP,
    compute_cross_head_cp,
)
from dflux.cp_surgeon import (
    CPSurgeon,
    CPSurgeonConfig,
    CPSurgeryResult,
)
from dflux.head_profiler import (
    HeadProfiler,
    HeadRole,
    ProfileReport,
    quick_profile,
    STIMULUS_BATTERIES,
)
from dflux.multiscale_telemetry import (
    MultiScaleTelemetry,
    TelemetryConfig,
    TokenSnapshot,
    quick_telemetry,
)

__all__ = [
    "DFluxMeter", "DFluxConfig",
    "AXEEngine", "AXEConfig", "Regime", "Action", "Witness",
    "JState", "classify_regime",
    "InferenceProbe", "ProbeConfig", "TokenDiagnostic",
    "FineProbe", "FineProbeConfig", "FineTokenDiagnostic",
    "HeadSurgeon", "HeadIntervention", "SurgeryReport",
    "CausalPrimitives", "CPConfig", "HeadCP", "compute_cross_head_cp",
    "CPSurgeon", "CPSurgeonConfig", "CPSurgeryResult",
    "HeadProfiler", "HeadRole", "ProfileReport",
    "quick_profile", "STIMULUS_BATTERIES",
    "MultiScaleTelemetry", "TelemetryConfig", "TokenSnapshot",
    "quick_telemetry",
    "__version__",
]
