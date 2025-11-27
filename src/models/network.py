from __future__ import annotations

from typing import Dict, Any, List

import torch
import torch.nn as nn
import yaml

from .blocks import (
    ConvNormAct1d,
    MambaBlock1D,
    KANBlock1D,
    DownSample1D,
    UpSample1D,
    ConvHead1D,
)


class SeisMambaKAN(nn.Module):
    """
    U-Net style hybrid Mamba + KAN network for seismic phase picking.

    Expected input:
        x: (B, C_in, T)  typically (B, 3, 6000)

    Outputs:
        {
            "detection":  (B, T),
            "p_gaussian": (B, T),
            "s_gaussian": (B, T),
        }

    Configuration format (YAML -> dict) is assumed to have:
        - "model":   global model-level options
        - "encoder": stem + stages + mamba sub-config
        - "decoder": stages + kan sub-config
        - "heads":   detection / p_gaussian / s_gaussian
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()

        # Store full config
        self.cfg = cfg

        model_cfg = cfg.get("model", {})
        enc_cfg = cfg.get("encoder", {})
        dec_cfg = cfg.get("decoder", {})
        heads_cfg = cfg.get("heads", {})

        # ------------------------------------------------------------------
        # Global model options
        # ------------------------------------------------------------------
        self.input_channels: int = int(model_cfg.get("input_channels", 3))
        self.input_length: int = int(model_cfg.get("input_length", 6000))
        self.norm_type: str = model_cfg.get("normalization", "layernorm")
        self.activation: str = model_cfg.get("activation", "silu")
        self.dropout: float = float(model_cfg.get("dropout", 0.1))

        # ------------------------------------------------------------------
        # Encoder: stem + stages (with Mamba)
        # ------------------------------------------------------------------
        stem_cfg = enc_cfg.get("stem", {})
        stem_out_channels = int(stem_cfg.get("out_channels", 16))
        stem_kernel = int(stem_cfg.get("kernel_size", 7))
        stem_stride = int(stem_cfg.get("stride", 1))
        stem_padding = int(stem_cfg.get("padding", 3))
        stem_bias = bool(stem_cfg.get("bias", True))

        self.stem = ConvNormAct1d(
            in_channels=self.input_channels,
            out_channels=stem_out_channels,
            kernel_size=stem_kernel,
            stride=stem_stride,
            padding=stem_padding,
            bias=stem_bias,
            norm_type=self.norm_type,
            activation=self.activation,
        )

        # Encoder stages specification
        encoder_stages_cfg: List[Dict[str, Any]] = enc_cfg.get("stages", [])

        # Mamba config
        mamba_cfg = enc_cfg.get("mamba", {})
        self.mamba_d_state: int = int(mamba_cfg.get("d_state", 16))
        self.mamba_expand: float = float(mamba_cfg.get("expand_factor", 2.0))
        self.mamba_conv_kernel: int = int(mamba_cfg.get("conv_kernel", 3))
        self.mamba_dropout: float = float(mamba_cfg.get("dropout", self.dropout))
        self.mamba_layer_norm_eps: float = float(mamba_cfg.get("layer_norm_eps", 1e-5))

        # We keep module dicts keyed by stage name
        self.encoder_blocks = nn.ModuleDict()
        self.encoder_stage_specs = encoder_stages_cfg  # to preserve order

        # Map encoder stage name -> channel dimension (after its blocks)
        self.encoder_channels: Dict[str, int] = {}

        prev_channels = stem_out_channels

        for stage_cfg in encoder_stages_cfg:
            name = stage_cfg["name"]
            d_model = int(stage_cfg["d_model"])
            n_layers = int(stage_cfg.get("n_layers", 1))
            down_factor = int(stage_cfg.get("downsample_factor", 1))
            use_mamba = bool(stage_cfg.get("use_mamba", True))

            blocks: List[nn.Module] = []

            # Optional downsampling (except for factor=1)
            if down_factor > 1:
                blocks.append(
                    DownSample1D(
                        in_channels=prev_channels,
                        out_channels=d_model,
                        factor=down_factor,
                        norm_type=self.norm_type,
                        activation=self.activation,
                    )
                )
            else:
                # If no downsampling but channel size changes, align channels via 1x1 conv
                if prev_channels != d_model:
                    blocks.append(
                        ConvNormAct1d(
                            in_channels=prev_channels,
                            out_channels=d_model,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=True,
                            norm_type=self.norm_type,
                            activation=self.activation,
                        )
                    )

            # Mamba blocks
            if use_mamba:
                for _ in range(n_layers):
                    blocks.append(
                        MambaBlock1D(
                            d_model=d_model,
                            d_state=self.mamba_d_state,
                            d_conv=self.mamba_conv_kernel,
                            expand=self.mamba_expand,
                            dropout=self.mamba_dropout,
                            norm_type=self.norm_type,
                            layer_norm_eps=self.mamba_layer_norm_eps,
                        )
                    )

            self.encoder_blocks[name] = nn.Sequential(*blocks)
            self.encoder_channels[name] = d_model
            prev_channels = d_model

        self.bottleneck_name = encoder_stages_cfg[-1]["name"] if encoder_stages_cfg else None

        # ------------------------------------------------------------------
        # Decoder: upsample + skip + KAN blocks
        # ------------------------------------------------------------------
        dec_stages_cfg: List[Dict[str, Any]] = dec_cfg.get("stages", [])
        kan_cfg = dec_cfg.get("kan", {})
        self.decoder_concat_skips: bool = bool(dec_cfg.get("concat_skips", True))
        self.upsample_mode: str = dec_cfg.get("upsample_mode", "conv_transpose")

        self.kan_grid_size: int = int(kan_cfg.get("grid_size", 3))
        self.kan_spline_order: int = int(kan_cfg.get("spline_order", 3))
        self.kan_base_activation: str = kan_cfg.get("base_activation", "silu")
        self.kan_dropout: float = float(kan_cfg.get("dropout", self.dropout))
        self.kan_layer_norm_eps: float = float(kan_cfg.get("layer_norm_eps", 1e-5))
        self.kan_enable_standalone_scale_spline: bool = bool(
            kan_cfg.get("enable_standalone_scale_spline", True)
        )

        self.decoder_stage_specs = dec_stages_cfg
        self.decoder_ups = nn.ModuleDict()
        self.decoder_proj = nn.ModuleDict()
        self.decoder_blocks = nn.ModuleDict()

        # Starting channels for decoder is the bottleneck d_model
        if self.bottleneck_name is None:
            raise ValueError("Encoder stages are empty; bottleneck not defined.")
        current_channels = self.encoder_channels[self.bottleneck_name]

        for stage_cfg in dec_stages_cfg:
            name = stage_cfg["name"]
            d_model = int(stage_cfg["d_model"])
            n_layers = int(stage_cfg.get("n_layers", 1))
            up_factor = int(stage_cfg.get("upsample_factor", 2))
            skip_from = stage_cfg.get("skip_from", None)

            # 1) Upsample: (B, current_channels, T) -> (B, d_model, T_up)
            self.decoder_ups[name] = UpSample1D(
                in_channels=current_channels,
                out_channels=d_model,
                factor=up_factor,
                norm_type=self.norm_type,
                activation=self.activation,
            )

            # 2) After concatenation with skip, channels increase
            in_after_concat = d_model
            if self.decoder_concat_skips and skip_from is not None:
                if skip_from not in self.encoder_channels:
                    raise KeyError(
                        f"Decoder stage '{name}' refers to unknown skip_from='{skip_from}'."
                    )
                in_after_concat += self.encoder_channels[skip_from]

            # 3) Project concatenated channels back to d_model
            self.decoder_proj[name] = ConvNormAct1d(
                in_channels=in_after_concat,
                out_channels=d_model,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                norm_type=self.norm_type,
                activation=self.activation,
            )

            # 4) KAN blocks
            blocks: List[nn.Module] = []
            for _ in range(n_layers):
                blocks.append(
                    KANBlock1D(
                        d_model=d_model,
                        grid_size=self.kan_grid_size,
                        spline_order=self.kan_spline_order,
                        base_activation=self.kan_base_activation,
                        dropout=self.kan_dropout,
                        norm_type=self.norm_type,
                        layer_norm_eps=self.kan_layer_norm_eps,
                        enable_standalone_scale_spline=self.kan_enable_standalone_scale_spline,
                    )
                )
            self.decoder_blocks[name] = nn.Sequential(*blocks)

            current_channels = d_model

        # ------------------------------------------------------------------
        # Heads: detection / P / S
        # ------------------------------------------------------------------
        shared_heads_cfg = heads_cfg.get("shared", {})
        heads_in_channels = int(shared_heads_cfg.get("in_channels", current_channels))

        if heads_in_channels != current_channels:
            # Optional 1x1 conv to align channels for head input
            self.head_input_proj = ConvNormAct1d(
                in_channels=current_channels,
                out_channels=heads_in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
                norm_type=self.norm_type,
                activation=self.activation,
            )
        else:
            self.head_input_proj = nn.Identity()

        # Detection head
        det_cfg = heads_cfg.get("detection", {})
        det_out_channels = int(det_cfg.get("out_channels", 1))
        det_activation = det_cfg.get("activation", "sigmoid")

        self.head_detection = ConvHead1D(
            in_channels=heads_in_channels,
            out_channels=det_out_channels,
            activation=det_activation,
            bias=bool(shared_heads_cfg.get("bias", True)),
        )

        # P-gaussian head
        p_cfg = heads_cfg.get("p_gaussian", {})
        p_out_channels = int(p_cfg.get("out_channels", 1))
        p_activation = p_cfg.get("activation", "none")

        self.head_p = ConvHead1D(
            in_channels=heads_in_channels,
            out_channels=p_out_channels,
            activation=p_activation,
            bias=bool(shared_heads_cfg.get("bias", True)),
        )

        # S-gaussian head
        s_cfg = heads_cfg.get("s_gaussian", {})
        s_out_channels = int(s_cfg.get("out_channels", 1))
        s_activation = s_cfg.get("activation", "none")

        self.head_s = ConvHead1D(
            in_channels=heads_in_channels,
            out_channels=s_out_channels,
            activation=s_activation,
            bias=bool(shared_heads_cfg.get("bias", True)),
        )

        # ------------------------------------------------------------------
        # Optional weight init
        # ------------------------------------------------------------------
        init_cfg = cfg.get("init", {})
        init_type = init_cfg.get("type", "xavier_uniform")
        init_gain = float(init_cfg.get("gain", 1.0))
        self._apply_init(init_type, init_gain)

    # ----------------------------------------------------------------------
    # Initialization helper
    # ----------------------------------------------------------------------
    def _apply_init(self, init_type: str, gain: float = 1.0) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                if init_type == "xavier_uniform":
                    nn.init.xavier_uniform_(m.weight, gain=gain)
                elif init_type == "xavier_normal":
                    nn.init.xavier_normal_(m.weight, gain=gain)
                elif init_type == "kaiming_uniform":
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                elif init_type == "kaiming_normal":
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                else:
                    # default PyTorch init
                    pass
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ----------------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (B, C_in, T) tensor, e.g. (B, 3, 6000)

        Returns:
            A dict with keys:
                - "detection":  (B, T)
                - "p_gaussian": (B, T)
                - "s_gaussian": (B, T)
        """
        if x.dim() != 3:
            raise ValueError(f"Expected input of shape (B, C, T), got {x.shape}")

        if x.size(1) != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channels, got {x.size(1)}."
            )

        # Stem
        x = self.stem(x)

        # Encoder
        skips: Dict[str, torch.Tensor] = {}

        for stage_cfg in self.encoder_stage_specs:
            name = stage_cfg["name"]
            x = self.encoder_blocks[name](x)
            # Save skip for all encoder stages except bottleneck
            if name != self.bottleneck_name:
                skips[name] = x

        # Bottleneck output is current x
        z = x

        # Decoder
        y = z
        for stage_cfg in self.decoder_stage_specs:
            name = stage_cfg["name"]
            skip_from = stage_cfg.get("skip_from", None)

            # 1) Upsample
            y = self.decoder_ups[name](y)

            # 2) Concat skip if enabled
            if self.decoder_concat_skips and skip_from is not None:
                if skip_from not in skips:
                    raise KeyError(
                        f"Skip feature '{skip_from}' not found in encoder outputs."
                    )
                skip_feat = skips[skip_from]
                if skip_feat.size(-1) != y.size(-1):
                    # In case of off-by-one due to padding/stride, center crop or pad
                    min_len = min(skip_feat.size(-1), y.size(-1))
                    skip_feat = skip_feat[..., :min_len]
                    y = y[..., :min_len]
                y = torch.cat([y, skip_feat], dim=1)

            # 3) Project to d_model
            y = self.decoder_proj[name](y)

            # 4) KAN blocks
            y = self.decoder_blocks[name](y)

        # Heads
        h = self.head_input_proj(y)

        det = self.head_detection(h).squeeze(1)  # (B, T)
        p = self.head_p(h).squeeze(1)            # (B, T)
        s = self.head_s(h).squeeze(1)            # (B, T)

        return {
            "detection": det,
            "p_gaussian": p,
            "s_gaussian": s,
        }

    # ----------------------------------------------------------------------
    # Utility constructors
    # ----------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str) -> "SeisMambaKAN":
        """
        Build model directly from a YAML config file that defines
        model / encoder / decoder / heads sections.
        """
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cls(cfg)

    def num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
