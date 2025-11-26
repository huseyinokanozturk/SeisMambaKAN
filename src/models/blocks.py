import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except ImportError as e:
    Mamba = None
    _mamba_import_error = e

try:
    from efficient_kan import KANLinear
except ImportError as e:
    KANLinear = None
    _kan_import_error = e


# ============================================================
# Helper: Activations & Normalizations
# ============================================================

def get_activation(name: str) -> nn.Module:
    """
    Returns an activation module given its name.
    Supported: 'relu', 'gelu', 'silu', 'tanh', 'none'
    """
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "silu" or name == "swish":
        return nn.SiLU(inplace=True)
    if name == "tanh":
        return nn.Tanh()
    if name == "none" or name is None:
        return nn.Identity()
    raise ValueError(f"Unknown activation: {name}")


class ChannelLayerNorm1d(nn.Module):
    """
    LayerNorm over channel dimension for tensors of shape (B, C, T).

    PyTorch's nn.LayerNorm expects the normalized dimension to be the last one,
    so we temporarily permute to (B, T, C), apply LayerNorm over C, and permute back.
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(num_features, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = x.transpose(1, 2)  # (B, T, C)
        x = self.ln(x)
        x = x.transpose(1, 2)  # (B, C, T)
        return x


def get_norm(norm_type: str, num_features: int, eps: float = 1e-5) -> nn.Module:
    """
    Returns a normalization layer for 1D sequences (B, C, T).
    norm_type: 'batchnorm', 'layernorm', or 'none'
    """
    if norm_type is None or norm_type.lower() == "none":
        return nn.Identity()
    norm_type = norm_type.lower()
    if norm_type in ("batchnorm", "batchnorm1d", "bn", "bn1d"):
        return nn.BatchNorm1d(num_features, eps=eps)
    if norm_type in ("layernorm", "ln"):
        return ChannelLayerNorm1d(num_features, eps=eps)
    raise ValueError(f"Unknown norm_type: {norm_type}")


# ============================================================
# Basic Building Block: Conv → Norm → Act
# ============================================================

class ConvNormAct1d(nn.Module):
    """
    Standard 1D conv block: Conv1d -> Norm -> Activation.

    Input:  (B, C_in,  T)
    Output: (B, C_out, T_out)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        bias: bool = True,
        norm_type: str = "layernorm",
        activation: str = "silu",
    ):
        super().__init__()
        if padding is None:
            # Same padding for stride=1; for stride>1 caller should set explicitly
            padding = (kernel_size - 1) // 2

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.norm = get_norm(norm_type, out_channels)
        self.act = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


# ============================================================
# MambaBlock1D: Residual Mamba block for encoder/bottleneck
# ============================================================

class MambaBlock1D(nn.Module):
    """
    Residual Mamba block operating on (B, C, T).

    Internally:
      - Normalize over channels
      - Permute to (B, T, C)
      - Apply Mamba(d_model=C, ...)
      - Permute back to (B, C, T)
      - Dropout + residual
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: float = 2.0,
        dropout: float = 0.0,
        norm_type: str = "layernorm",
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()

        if Mamba is None:
            raise ImportError(
                "Could not import Mamba from mamba_ssm. "
                "Please install it via `pip install mamba-ssm`."
            ) from _mamba_import_error

        self.d_model = d_model
        self.norm = get_norm(norm_type, d_model, eps=layer_norm_eps)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        """
        residual = x
        x = self.norm(x)              # (B, C, T)
        x = x.transpose(1, 2)         # (B, T, C)
        x = self.mamba(x)             # (B, T, C)
        x = x.transpose(1, 2)         # (B, C, T)
        x = self.dropout(x)
        return x + residual


# ============================================================
# KANBlock1D: Residual KAN block for decoder
# ============================================================

class KANBlock1D(nn.Module):
    """
    Residual KAN block for 1D sequences.

    - Takes (B, C, T).
    - Normalizes over channels.
    - Treats last dim as feature dim via (B, T, C).
    - Applies a KANLinear(C -> C) over the feature dimension.
    - Residual connection in (B, C, T) space.
    """

    def __init__(
        self,
        d_model: int,
        grid_size: int = 3,
        spline_order: int = 3,
        base_activation: str = "silu",
        dropout: float = 0.0,
        norm_type: str = "layernorm",
        layer_norm_eps: float = 1e-5,
        enable_standalone_scale_spline: bool = True,
    ):
        super().__init__()

        if KANLinear is None:
            raise ImportError(
                "Could not import KANLinear from efficient_kan. "
                "Please install it via `pip install efficient-kan`."
            ) from _kan_import_error

        self.d_model = d_model
        self.norm = get_norm(norm_type, d_model, eps=layer_norm_eps)

        # KANLinear works on the last dimension; we will feed (B, T, C)
        self.kan = KANLinear(
            in_features=d_model,
            out_features=d_model,
            grid_size=grid_size,
            spline_order=spline_order,
            base_activation=base_activation,
            enable_standalone_scale_spline=enable_standalone_scale_spline,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        """
        residual = x
        x = self.norm(x)              # (B, C, T)
        x = x.transpose(1, 2)         # (B, T, C)
        x = self.kan(x)               # (B, T, C)
        x = self.dropout(x)
        x = x.transpose(1, 2)         # (B, C, T)
        return x + residual


# ============================================================
# DownSample1D: for encoder
# ============================================================

class DownSample1D(nn.Module):
    """
    Downsampling block for 1D signals.

    Default: Conv1d with stride = factor (typically 2).
    Input:  (B, C_in,  T)
    Output: (B, C_out, T / factor)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int = 2,
        kernel_size: int = 4,
        bias: bool = True,
        norm_type: str = "layernorm",
        activation: str = "silu",
    ):
        super().__init__()

        # For stride=factor, kernel_size=4, a padding=kernel_size//2 - 1 works well
        # to avoid shrinking too much. For factor=2, kernel=4, padding=1.
        stride = factor
        padding = kernel_size // 2 - 1

        self.block = ConvNormAct1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            norm_type=norm_type,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ============================================================
# UpSample1D: for decoder
# ============================================================

class UpSample1D(nn.Module):
    """
    Upsampling block for 1D signals using ConvTranspose1d.

    Default: kernel_size=factor, stride=factor
    For factor=2, kernel=2, stride=2 → exact x2 length.

    Input:  (B, C_in,  T)
    Output: (B, C_out, T * factor)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int = 2,
        kernel_size: Optional[int] = None,
        bias: bool = True,
        norm_type: str = "layernorm",
        activation: str = "silu",
    ):
        super().__init__()
        if kernel_size is None:
            kernel_size = factor

        # For stride=factor, kernel=factor, padding=0 → L_out = (L_in - 1)*factor + kernel
        # If kernel=factor, this becomes L_out = (L_in - 1)*factor + factor = L_in * factor
        stride = factor
        padding = 0

        self.conv_transpose = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.norm = get_norm(norm_type, out_channels)
        self.act = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_transpose(x)
        x = self.norm(x)
        x = self.act(x)
        return x


# ============================================================
# ConvHead1D: generic prediction head
# ============================================================

class ConvHead1D(nn.Module):
    """
    Simple 1D head:
      - 1x1 Conv (C_in -> C_out)
      - optional activation

    Input:  (B, C_in,  T)
    Output: (B, C_out, T)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "none",
        bias: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias,
        )
        self.act = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.act(x)
        return x
