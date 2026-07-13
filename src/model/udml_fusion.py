import torch
import torch.nn as nn
import torch.nn.functional as F


class UDMLFusion(nn.Module):
    """Uncertainty- and Dependency-aware dynamic fusion (UDML).

    Faithful re-implementation of ``AVClassifier_AUXI_UDML``
    (shicaiwei123/UDML) for sagittal/axial visual tokens:

      (1) Uncertainty   sigma_m = exp(0.5 * estimator(feat_m.detach())),
          supervised by MSE against the injected-noise variance labels
          (see :meth:`uncertainty_loss`).
      (2) Inverse-variance target weight -- the less uncertain modality gets
          a larger weight:
              w~_sag = 2 * sigma_ax^2  / (sigma_sag^2 + sigma_ax^2)
              w~_ax  = 2 * sigma_sag^2 / (sigma_sag^2 + sigma_ax^2)
      (3) Dependency d_m -- how strongly the model already relies on modality
          m ALONE.  In the paper this is ``E[ sum_c |out_m| ]`` measured on the
          unimodal heads, detached, and reused on the next step.  Here it is an
          EMA buffer fed from the unimodal LM outputs via
          :meth:`update_dependency`.
      (4) Debias by DIVIDING the target weight by the dependency, then
          renormalize so the two weights sum to 2:
              w_m = w~_m / d_m ;   w_m <- 2 * w_m / (w_sag + w_ax)
      (5) Fuse, centered at the plain average (weights average to 1):
              final = 0.5 * (w_sag * feat_sag + w_ax * feat_ax)

    Notes vs. the reference:
      * The reference divides by the dependency (it suppresses the
        over-relied-upon modality to force balanced learning).  An earlier
        version of this module *multiplied* by a leave-one-out contribution,
        which is the opposite behaviour -- that is fixed here.
      * ``update_dependency`` is called from the LM after the unimodal forward
        passes (the same outputs that produce the unimodal auxiliary CE loss),
        mirroring how the reference measures dependency from ``out_a``/``out_v``.
      * Absolute dependency scale cancels in the renormalization step, so only
        the *ratio* d_sag : d_ax matters -- exactly as in the reference.
    """

    def __init__(self, hidden_size, var_loss_weight=0.1, eps=1e-6,
                 warmup_steps=0, depend_momentum=0.9):
        super().__init__()
        estimator_hidden = max(hidden_size // 2, 1)
        self.sag_variance_estimator = nn.Sequential(
            nn.Linear(hidden_size, estimator_hidden),
            nn.Dropout(0.1),
            nn.Linear(estimator_hidden, 1),
        )
        self.ax_variance_estimator = nn.Sequential(
            nn.Linear(hidden_size, estimator_hidden),
            nn.Dropout(0.1),
            nn.Linear(estimator_hidden, 1),
        )
        # Learnable fusion backbone (as in the reference: the UDML weight is a
        # *modulation* on top of a real fusion module, not the whole fusion).
        # Reference: a_out, v_out, out = fusion_module(a + w_a*a, v + w_v*v).
        from .base_fusion import GateFusion
        self.fusion_backbone = GateFusion(hidden_size)
        self.var_loss_weight = var_loss_weight
        self.eps = eps
        self.warmup_steps = int(warmup_steps)
        self.depend_momentum = float(depend_momentum)
        # Dependency d_m: a measured constant, EMA-updated from the unimodal
        # heads. Persisted so it carries across stage1 -> stage2.
        self.register_buffer("sag_depend", torch.ones(()))
        self.register_buffer("ax_depend", torch.ones(()))
        # Step counter for the warm-up gate; not part of the model weights.
        self.register_buffer("_train_steps", torch.zeros((), dtype=torch.long),
                             persistent=False)
        self.last_aux = {}

    def _estimate_scales(self, feat_sag, feat_ax):
        sag_pool = feat_sag.mean(dim=1)
        ax_pool = feat_ax.mean(dim=1)

        sag_scale = (self.sag_variance_estimator(sag_pool.detach()) * 0.5).exp() + self.eps
        ax_scale = (self.ax_variance_estimator(ax_pool.detach()) * 0.5).exp() + self.eps
        return sag_pool, ax_pool, sag_scale, ax_scale

    def uncertainty_loss(self, feat_sag, feat_ax, sag_variance, ax_variance):
        _, _, sag_scale, ax_scale = self._estimate_scales(feat_sag, feat_ax)
        sag_target = sag_variance.to(device=feat_sag.device, dtype=sag_scale.dtype).view_as(sag_scale)
        ax_target = ax_variance.to(device=feat_ax.device, dtype=ax_scale.dtype).view_as(ax_scale)
        return (
            F.mse_loss(sag_scale, sag_target) + F.mse_loss(ax_scale, ax_target)
        ) * self.var_loss_weight

    @torch.no_grad()
    def update_dependency(self, sag_value, ax_value):
        """EMA-update the (detached) per-modality dependency.

        Mirrors the reference, where ``args.audio_depend = mean(|out_a|)`` is
        measured each step from the unimodal output and divided into the fusion
        weight on the following step.
        """
        m = self.depend_momentum
        sag_value = torch.as_tensor(sag_value, dtype=self.sag_depend.dtype,
                                    device=self.sag_depend.device).detach()
        ax_value = torch.as_tensor(ax_value, dtype=self.ax_depend.dtype,
                                   device=self.ax_depend.device).detach()
        self.sag_depend.mul_(m).add_(sag_value * (1.0 - m))
        self.ax_depend.mul_(m).add_(ax_value * (1.0 - m))

    def forward(self, feat_sag, feat_ax):
        sag_pool, ax_pool, sag_scale, ax_scale = self._estimate_scales(feat_sag, feat_ax)

        # (2) inverse-variance target weight: lower uncertainty -> higher weight.
        denom = sag_scale.square() + ax_scale.square() + self.eps
        target_weight_sag = 2.0 * ax_scale.square() / denom
        target_weight_ax = 2.0 * sag_scale.square() / denom

        # (4) debias: DIVIDE by the (measured, detached) dependency, then
        # renormalize so the two weights sum to 2. Only the ratio matters.
        weight_sag = target_weight_sag / (self.sag_depend + self.eps)
        weight_ax = target_weight_ax / (self.ax_depend + self.eps)
        weight_denom = weight_sag + weight_ax + self.eps
        weight_sag = 2.0 * weight_sag / weight_denom
        weight_ax = 2.0 * weight_ax / weight_denom

        # Warm-up: keep the plain average while the variance estimator is still
        # unreliable (the reference forces weight=1,1 for the first epochs).
        if self.training and self.warmup_steps > 0:
            if int(self._train_steps) < self.warmup_steps:
                weight_sag = torch.ones_like(weight_sag)
                weight_ax = torch.ones_like(weight_ax)
            self._train_steps += 1

        # (5) Apply the weight as a RESIDUAL modulation (reference: a + w_a*a =
        # (1+w_a)*a), then fuse through the learnable backbone. This keeps the
        # base feature and lets a real fusion module combine the two streams,
        # instead of a bare weighted average.
        mod_sag = (1.0 + weight_sag).unsqueeze(1) * feat_sag
        mod_ax = (1.0 + weight_ax).unsqueeze(1) * feat_ax
        final_feat = self.fusion_backbone(mod_sag, mod_ax)

        self.last_aux = {
            "loss": feat_sag.new_zeros(()),
            "sag_scale": sag_scale.detach(),
            "ax_scale": ax_scale.detach(),
            "weight_sag": weight_sag.detach(),
            "weight_ax": weight_ax.detach(),
            "sag_depend": self.sag_depend.detach().clone(),
            "ax_depend": self.ax_depend.detach().clone(),
        }
        return final_feat
