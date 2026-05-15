import torch
import torch.nn as nn
import torch.nn.functional as F


class UDMLFusion(nn.Module):
    """UDML-inspired dynamic fusion for sagittal and axial visual tokens."""

    def __init__(self, hidden_size, var_loss_weight=0.1, aux_dim=None, eps=1e-6):
        super().__init__()
        estimator_hidden = max(hidden_size // 2, 1)
        aux_dim = aux_dim or hidden_size
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
        self.shared_aux_head = nn.Linear(hidden_size * 2, aux_dim)
        self.var_loss_weight = var_loss_weight
        self.eps = eps
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

    def forward(self, feat_sag, feat_ax):
        sag_pool, ax_pool, sag_scale, ax_scale = self._estimate_scales(feat_sag, feat_ax)

        denom = sag_scale.square() + ax_scale.square() + self.eps
        target_weight_sag = 2.0 * ax_scale.square() / denom
        target_weight_ax = 2.0 * sag_scale.square() / denom

        zeros_sag = torch.zeros_like(sag_pool)
        zeros_ax = torch.zeros_like(ax_pool)
        full_aux = self.shared_aux_head(torch.cat([sag_pool, ax_pool], dim=-1))
        sag_aux = self.shared_aux_head(torch.cat([sag_pool, zeros_ax], dim=-1))
        ax_aux = self.shared_aux_head(torch.cat([zeros_sag, ax_pool], dim=-1))

        sag_depend = sag_aux.abs().mean(dim=0, keepdim=True).sum(dim=1, keepdim=True) + self.eps
        ax_depend = ax_aux.abs().mean(dim=0, keepdim=True).sum(dim=1, keepdim=True) + self.eps

        weight_sag = target_weight_sag / sag_depend
        weight_ax = target_weight_ax / ax_depend
        weight_denom = weight_sag + weight_ax + self.eps
        weight_sag = 2.0 * weight_sag / weight_denom
        weight_ax = 2.0 * weight_ax / weight_denom

        final_feat = 0.5 * (
            weight_sag.unsqueeze(1) * feat_sag + weight_ax.unsqueeze(1) * feat_ax
        )

        self.last_aux = {
            "loss": feat_sag.new_zeros(()),
            "sag_scale": sag_scale.detach(),
            "ax_scale": ax_scale.detach(),
            "weight_sag": weight_sag.detach(),
            "weight_ax": weight_ax.detach(),
            "sag_depend": sag_depend.detach(),
            "ax_depend": ax_depend.detach(),
            "full_aux": full_aux.detach(),
            "sag_aux": sag_aux.detach(),
            "ax_aux": ax_aux.detach(),
        }
        return final_feat
