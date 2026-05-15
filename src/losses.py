from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F



class SeisMambaKANLoss(nn.Module):
    """
    Multi-head loss for SeisMambaKAN.

    This module implements:
      - Detection loss: weighted BCE (optional focal variant) on detection head.
      - Phase loss for P and S: detection-masked, peak-weighted MSE on Gaussian targets.

    The behavior is fully controlled by the "loss" section of the main config.yaml.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()

        loss_cfg = cfg.get("loss", {})
        if not loss_cfg:
            raise ValueError("Missing 'loss' section in config.")

        # Keys mapping: how model outputs align with label dictionary
        keys_cfg = loss_cfg.get("keys", {})
        if not keys_cfg:
            raise ValueError("Missing 'loss.keys' section in config.")

        # Output keys are fixed by network.py
        self.out_key_det: str = "detection"
        self.out_key_p: str = "p_gaussian"
        self.out_key_s: str = "s_gaussian"

        # Label keys are configurable via config.yaml
        self.label_key_det: str = str(keys_cfg.get("detection", "y_det"))
        self.label_key_p: str = str(keys_cfg.get("p", "y_p"))
        self.label_key_s: str = str(keys_cfg.get("s", "y_s"))

        # Global head weights: L_total = w_det * L_det + w_p * L_p + w_s * L_s
        weights_cfg = loss_cfg.get("weights", {})
        self.w_det: float = float(weights_cfg.get("detection", 1.0))
        self.w_p: float = float(weights_cfg.get("p", 1.0))
        self.w_s: float = float(weights_cfg.get("s", 1.0))

        # Detection loss configuration
        det_cfg = loss_cfg.get("detection", {})
        self.det_loss_type: str = str(det_cfg.get("loss_type", "bce")).lower()
        self.det_positive_weight: float = float(det_cfg.get("positive_weight", 1.0))
        self.det_use_focal: bool = bool(det_cfg.get("use_focal", False))
        self.det_focal_alpha: float = float(det_cfg.get("focal_alpha", 0.25))
        self.det_focal_gamma: float = float(det_cfg.get("focal_gamma", 2.0))
        self.det_eps: float = float(det_cfg.get("eps", 1.0e-7))

        # Phase loss configuration (shared for P and S)
        phase_cfg = loss_cfg.get("phase", {})
        self.phase_loss_type: str = str(
            phase_cfg.get("loss_type", "masked_peak_mse")
        ).lower()
        self.phase_peak_weight_scale: float = float(
            phase_cfg.get("peak_weight_scale", 0.0)
        )
        self.phase_use_det_mask: bool = bool(
            phase_cfg.get("use_detection_mask", True)
        )
        self.phase_normalize_by_mask: bool = bool(
            phase_cfg.get("normalize_by_mask", True)
        )
        self.phase_clamp_predictions: bool = bool(
            phase_cfg.get("clamp_predictions", True)
        )
        self.phase_clamp_min: float = float(phase_cfg.get("clamp_min", 0.0))
        self.phase_clamp_max: float = float(phase_cfg.get("clamp_max", 1.0))
        self.phase_eps: float = float(phase_cfg.get("eps", 1.0e-6))

        # Optional arrival-time based center loss (soft-argmax over time)
        center_cfg = loss_cfg.get("center_loss", {})
        self.center_enabled: bool = bool(center_cfg.get("enabled", False))
        self.center_weight: float = float(center_cfg.get("weight", 0.0))
        self.center_temperature: float = float(center_cfg.get("temperature", 10.0))

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total and per-head losses.

        Args:
            outputs: dict from the model forward:
                {
                    "detection":  (B, T),
                    "p_gaussian": (B, T),
                    "s_gaussian": (B, T),
                }
            labels: dict from dataset:
                at least the keys defined in self.label_key_*.

        Returns:
            A dict of scalar losses:
                {
                    "total":      L_total,
                    "detection":  L_det,
                    "p":          L_p,
                    "s":          L_s,
                    # optionally "center_p", "center_s" if center loss enabled
                }
        """
        # Resolve device from any prediction tensor
        device = self._infer_device(outputs)

        # ------------------------------------------------------------------
        # Fetch predictions
        # ------------------------------------------------------------------
        det_pred = outputs[self.out_key_det].to(device)       # (B, T)
        p_pred = outputs[self.out_key_p].to(device)           # (B, T)
        s_pred = outputs[self.out_key_s].to(device)           # (B, T)

        # ------------------------------------------------------------------
        # Fetch targets
        # ------------------------------------------------------------------
        if self.label_key_det not in labels:
            raise KeyError(f"Detection label key '{self.label_key_det}' not found in labels.")
        if self.label_key_p not in labels:
            raise KeyError(f"P-phase label key '{self.label_key_p}' not found in labels.")
        if self.label_key_s not in labels:
            raise KeyError(f"S-phase label key '{self.label_key_s}' not found in labels.")

        det_target = labels[self.label_key_det].to(device).float()  # (B, T)
        p_target = labels[self.label_key_p].to(device).float()      # (B, T)
        s_target = labels[self.label_key_s].to(device).float()      # (B, T)

        # Sanity check shapes
        self._check_same_shape(det_pred, det_target, "detection")
        self._check_same_shape(p_pred, p_target, "p_gaussian")
        self._check_same_shape(s_pred, s_target, "s_gaussian")

        # ------------------------------------------------------------------
        # Detection loss
        # ------------------------------------------------------------------
        loss_det = self._compute_detection_loss(det_pred, det_target)

        # ------------------------------------------------------------------
        # Phase losses (P and S)
        # ------------------------------------------------------------------
        loss_p = self._compute_phase_loss(
            phase_pred=p_pred,
            phase_target=p_target,
            det_target=det_target,
        )
        loss_s = self._compute_phase_loss(
            phase_pred=s_pred,
            phase_target=s_target,
            det_target=det_target,
        )

        # ------------------------------------------------------------------
        # Optional center-based arrival time loss
        # ------------------------------------------------------------------
        center_p = None
        center_s = None
        if self.center_enabled and self.center_weight > 0.0:
            center_p = self._compute_center_loss(p_pred, p_target)
            center_s = self._compute_center_loss(s_pred, s_target)
            loss_p = loss_p + self.center_weight * center_p
            loss_s = loss_s + self.center_weight * center_s

        # ------------------------------------------------------------------
        # Total weighted loss
        # ------------------------------------------------------------------
        total = (
            self.w_det * loss_det
            + self.w_p * loss_p
            + self.w_s * loss_s
        )

        out = {
            "total": total,
            "detection": loss_det,
            "p": loss_p,
            "s": loss_s,
        }
        if center_p is not None:
            out["center_p"] = center_p
        if center_s is not None:
            out["center_s"] = center_s

        return out

    # --------------------------------------------------------------------- #
    # Detection loss (BCEWithLogits / focal BCE)
    # --------------------------------------------------------------------- #
    def _compute_detection_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute detection loss.

        Convention (Phase 0): the detection head outputs LOGITS (no sigmoid).
        We use F.binary_cross_entropy_with_logits, which is AMP-safe and
        numerically stable via the log-sum-exp trick — no autocast disable
        needed, no eps clamp.

        Focal variant computes pt = sigmoid(logits) * target + (1-sigmoid) * (1-target),
        then scales BCE by alpha * (1-pt)**gamma.
        """
        # pos_weight: scalar broadcast tensor for class imbalance (y=1 weighting)
        pos_weight = None
        if self.det_positive_weight != 1.0 and self.det_positive_weight > 0.0:
            pos_weight = torch.tensor(
                self.det_positive_weight,
                device=pred.device,
                dtype=pred.dtype,
            )

        # Per-element BCE on logits
        bce = F.binary_cross_entropy_with_logits(
            pred,
            target,
            reduction="none",
            pos_weight=pos_weight,
        )

        # Optional focal modulation
        if self.det_use_focal or self.det_loss_type == "focal_bce":
            probs = torch.sigmoid(pred)
            pt = torch.where(target > 0.5, probs, 1.0 - probs)
            focal_factor = (1.0 - pt) ** self.det_focal_gamma
            bce = self.det_focal_alpha * focal_factor * bce

        return bce.mean()




    # --------------------------------------------------------------------- #
    # Phase loss: detection-masked, peak-weighted MSE
    # --------------------------------------------------------------------- #
    def _compute_phase_loss(
        self,
        phase_pred: torch.Tensor,
        phase_target: torch.Tensor,
        det_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute phase loss for one head (P or S).

        Strategy:
          - Optionally clamp predictions to [clamp_min, clamp_max].
          - Compute squared error (MSE) over time.
          - Weight errors with:
                w_t = (1 + peak_weight_scale * phase_target[t])
                if peak_weight_scale > 0, otherwise w_t = 1.
          - Optionally multiply by detection mask to focus on event window.
          - Per-sample mask normalization: each sample contributes equally to
            the final loss when it has an event window; pure-noise samples
            contribute exactly zero (no gradient on the phase head from noise).

        Phase 1.1 fix:
            The previous implementation normalized by the BATCH-wide weight
            sum and, when that sum was ~0 (e.g. a fully-noise batch), fell
            back to sq_error.mean() — which is the un-masked mean over the
            entire (B, T) tensor. That gave spurious phase-head gradient on
            noise samples and dominated mixed batches when the mask sum was
            small. The new implementation normalizes per-sample and averages
            only over event-bearing samples.
        """
        if self.phase_loss_type not in {"masked_peak_mse", "mse"}:
            raise ValueError(f"Unsupported phase loss_type: {self.phase_loss_type}")

        # Optionally clamp predictions to [0, 1]
        if self.phase_clamp_predictions:
            phase_pred = phase_pred.clamp(self.phase_clamp_min, self.phase_clamp_max)

        # Squared error (B, T)
        sq_error = (phase_pred - phase_target) ** 2

        # Peak weighting based on target amplitude (B, T)
        if self.phase_peak_weight_scale > 0.0:
            weights = 1.0 + self.phase_peak_weight_scale * phase_target
        else:
            weights = torch.ones_like(phase_target)

        # Optional detection mask (B, T)
        if self.phase_use_det_mask:
            weights = weights * det_target

        if not self.phase_normalize_by_mask:
            # Simple unnormalized mean (rare path; kept for backward compatibility).
            return (sq_error * weights).mean()

        # ----- Per-sample normalization -----
        # Numerator and denominator per sample (B,)
        num_per_sample = (sq_error * weights).sum(dim=-1)
        denom_per_sample = weights.sum(dim=-1)

        event_mask = denom_per_sample > self.phase_eps           # (B,) bool
        n_event = event_mask.sum()

        if int(n_event.item()) == 0:
            # All-noise batch: phase head receives no gradient from this term.
            return torch.zeros(
                (), device=phase_pred.device, dtype=phase_pred.dtype
            )

        # Avoid division by ~0 on noise samples; their numerator is 0 so the
        # result is masked back to 0 by the where().
        per_sample_loss = torch.where(
            event_mask,
            num_per_sample / (denom_per_sample + self.phase_eps),
            torch.zeros_like(num_per_sample),
        )
        return per_sample_loss.sum() / n_event.to(per_sample_loss.dtype)

    # --------------------------------------------------------------------- #
    # Optional center-based arrival time loss (soft-argmax)
    # --------------------------------------------------------------------- #
    def _compute_center_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute arrival-time aware loss using soft-argmax over the time axis.

        This is designed to be an optional term that directly penalizes
        shifts between the predicted and target peak locations.

        NOTE:
            This function assumes that 'target' has a single dominant peak
            (Gaussian-like label). If the label is flat or all zeros, the
            loss will naturally be small or uninformative.
        """
        if self.center_temperature <= 0.0:
            raise ValueError("center_temperature must be positive when center loss is enabled.")

        B, T = pred.shape

        # Convert inputs to logits for softmax; scale by temperature
        pred_logits = pred / self.center_temperature
        target_logits = target / self.center_temperature

        pred_weights = torch.softmax(pred_logits, dim=-1)       # (B, T)
        target_weights = torch.softmax(target_logits, dim=-1)   # (B, T)

        # Time indices: 0..T-1, broadcasted to (B, T)
        time_indices = torch.arange(T, device=pred.device, dtype=pred.dtype)
        time_indices = time_indices.unsqueeze(0).expand(B, -1)

        pred_center = (pred_weights * time_indices).sum(dim=-1)      # (B,)
        target_center = (target_weights * time_indices).sum(dim=-1)  # (B,)

        # Use L1 distance between centers (can be switched to Huber if desired)
        center_diff = torch.abs(pred_center - target_center)  # (B,)
        return center_diff.mean()

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _infer_device(outputs: Dict[str, torch.Tensor]) -> torch.device:
        """
        Infer device from any tensor in the outputs dict.
        """
        for v in outputs.values():
            if isinstance(v, torch.Tensor):
                return v.device
        # Fallback to CPU if no tensor found (should not happen in practice)
        return torch.device("cpu")

    @staticmethod
    def _check_same_shape(
        a: torch.Tensor,
        b: torch.Tensor,
        name: str,
    ) -> None:
        if a.shape != b.shape:
            raise ValueError(
                f"Shape mismatch for '{name}': pred shape {a.shape}, target shape {b.shape}"
            )


def build_loss_fn(cfg: Dict[str, Any]) -> SeisMambaKANLoss:
    """
    Factory function to construct the SeisMambaKAN loss module
    from a full configuration dictionary (loaded from config.yaml).

    Usage:
        cfg = yaml.safe_load(open("config.yaml"))
        loss_fn = build_loss_fn(cfg)
        loss_dict = loss_fn(outputs, labels)
    """
    return SeisMambaKANLoss(cfg)
