"""Baseline fusion modules for the sagittal/axial dual-encoder.

These are simple, well-known multimodal fusion strategies meant to serve as
*baselines* against :class:`~src.model.udml_fusion.UDMLFusion`.  Every module
here is a drop-in replacement for ``UDMLFusion``:

  * ``forward(feat_sag, feat_ax)`` takes two token sequences of shape
    ``[B, N, H]`` (H == ``mm_hidden_size``) and returns the fused tokens with
    the *same* shape ``[B, N, H]`` so the downstream ``mm_projector`` is
    unchanged.
  * The auxiliary hooks the rest of the codebase may call
    (``uncertainty_loss`` and ``update_dependency``) are provided as no-ops via
    :class:`_BaseFusion`, so the noise/dependency machinery of UDML simply does
    nothing for these baselines.

Available types (see :func:`build_fusion`):
    "elementwise" : element-wise add / mul / mean / max (parameter-free by
                    default, optional learnable per-channel gate).
    "concat"      : channel concat [B,N,2H] -> Linear(2H, H).
    "gate"        : per-channel sigmoid gate g; out = g*sag + (1-g)*ax.
    "bilinear"    : low-rank (factorized) bilinear pooling, MFB-style.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "ElementWiseFusion",
    "ConcatFusion",
    "GateFusion",
    "BilinearFusion",
    "build_fusion",
]


class _BaseFusion(nn.Module):
    """Common no-op hooks so baselines match the UDMLFusion call surface."""

    def __init__(self):
        super().__init__()
        # Mirrors UDMLFusion.last_aux; kept empty-ish for logging compatibility.
        self.last_aux = {}

    def uncertainty_loss(self, feat_sag, feat_ax, sag_variance, ax_variance):
        """No uncertainty supervision for baselines -> zero loss.

        Returns a scalar tensor on the right device/dtype so callers can add it
        unconditionally.
        """
        return feat_sag.new_zeros(())

    @torch.no_grad()
    def update_dependency(self, sag_value, ax_value):
        """No dependency debiasing for baselines -> no-op."""
        return None


class ElementWiseFusion(_BaseFusion):
    """Element-wise fusion: combine the two streams without changing channels.

    ``op``:
        "add"  -> feat_sag + feat_ax            (default)
        "mean" -> 0.5 * (feat_sag + feat_ax)
        "mul"  -> feat_sag * feat_ax            (Hadamard product)
        "max"  -> elementwise max

    With ``learnable=True`` a per-channel weight ``alpha`` blends the streams as
    ``alpha * feat_sag + (1 - alpha) * feat_ax`` (only meaningful for additive
    ops); it is initialised to an even 0.5 split.
    """

    def __init__(self, hidden_size, op="add", learnable=False):
        super().__init__()
        op = op.lower()
        if op not in ("add", "mean", "mul", "max"):
            raise ValueError(f"ElementWiseFusion: unknown op '{op}'")
        self.op = op
        self.learnable = bool(learnable)
        if self.learnable:
            # alpha in (0,1) via sigmoid; start at 0.5 (logit 0).
            self.alpha_logit = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, feat_sag, feat_ax):
        if self.learnable and self.op in ("add", "mean"):
            # convex per-channel blend; replaces the plain add/mean.
            alpha = torch.sigmoid(self.alpha_logit)
            return alpha * feat_sag + (1.0 - alpha) * feat_ax
        if self.op == "add":
            return feat_sag + feat_ax
        if self.op == "mean":
            return 0.5 * (feat_sag + feat_ax)
        if self.op == "mul":
            return feat_sag * feat_ax
        return torch.maximum(feat_sag, feat_ax)


class ConcatFusion(_BaseFusion):
    """Concatenate along the channel axis, then project back to ``hidden_size``.

    [B, N, H] + [B, N, H] -> concat [B, N, 2H] -> Linear(2H, H).
    """

    def __init__(self, hidden_size, dropout=0.0, activation=True):
        super().__init__()
        self.proj = nn.Linear(2 * hidden_size, hidden_size)
        self.act = nn.GELU() if activation else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, feat_sag, feat_ax):
        fused = torch.cat([feat_sag, feat_ax], dim=-1)
        fused = self.proj(fused)
        return self.dropout(self.act(fused))


class GateFusion(_BaseFusion):
    """Gated fusion with a per-channel sigmoid gate.

    g = sigmoid(W [feat_sag ; feat_ax] + b)  in [0,1]^H, computed per token.
    out = g * feat_sag + (1 - g) * feat_ax.

    The gate decides, channel-by-channel and token-by-token, how much of each
    modality to keep -- a learned, content-dependent version of the convex
    blend.
    """

    def __init__(self, hidden_size, dropout=0.0):
        super().__init__()
        self.gate = nn.Linear(2 * hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, feat_sag, feat_ax):
        gate_in = torch.cat([feat_sag, feat_ax], dim=-1)
        gate = torch.sigmoid(self.gate(gate_in))
        fused = gate * feat_sag + (1.0 - gate) * feat_ax
        return self.dropout(fused)


class BilinearFusion(_BaseFusion):
    """Low-rank (factorized) bilinear pooling, MFB-style.

    A full bilinear interaction ``feat_sag^T W feat_ax`` over H channels needs an
    H*H*H tensor, which is intractable for H=768.  We use the standard low-rank
    factorisation: project each stream to a ``rank``-dim space, take the
    element-wise (Hadamard) product to model the multiplicative interaction, then
    project back to ``hidden_size``.

        z = (U feat_sag) * (V feat_ax)      # [B, N, rank]
        out = P( dropout(z) )               # [B, N, H]

    Optional signed square-root + L2 power normalisation (as in MFB) stabilises
    the product's dynamic range.
    """

    def __init__(self, hidden_size, rank=256, dropout=0.1, power_norm=True):
        super().__init__()
        self.sag_proj = nn.Linear(hidden_size, rank)
        self.ax_proj = nn.Linear(hidden_size, rank)
        self.out_proj = nn.Linear(rank, hidden_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.power_norm = bool(power_norm)

    def forward(self, feat_sag, feat_ax):
        z = self.sag_proj(feat_sag) * self.ax_proj(feat_ax)  # [B, N, rank]
        z = self.dropout(z)
        if self.power_norm:
            # signed sqrt then L2 normalise along the feature axis (MFB).
            z = torch.sign(z) * torch.sqrt(torch.abs(z) + 1e-12)
            z = F.normalize(z, p=2, dim=-1)
        return self.out_proj(z)


_FUSION_REGISTRY = {
    "elementwise": ElementWiseFusion,
    "element_wise": ElementWiseFusion,
    "concat": ConcatFusion,
    "concatenation": ConcatFusion,
    "gate": GateFusion,
    "gated": GateFusion,
    "bilinear": BilinearFusion,
}


def build_fusion(fusion_type, hidden_size, **kwargs):
    """Factory: return a baseline fusion module by name.

    Args:
        fusion_type: one of {"elementwise", "concat", "gate", "bilinear"}
            (a few aliases are accepted).
        hidden_size: channel dim H of the visual tokens (== mm_hidden_size).
        **kwargs: forwarded to the chosen module's constructor (e.g. ``op`` for
            elementwise, ``rank`` for bilinear).

    Returns:
        nn.Module exposing ``forward(feat_sag, feat_ax) -> [B, N, H]`` plus the
        ``uncertainty_loss`` / ``update_dependency`` no-op hooks.
    """
    key = str(fusion_type).lower()
    if key not in _FUSION_REGISTRY:
        raise ValueError(
            f"Unknown fusion_type '{fusion_type}'. "
            f"Available: {sorted(set(_FUSION_REGISTRY))}"
        )
    return _FUSION_REGISTRY[key](hidden_size, **kwargs)
