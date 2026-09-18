"""
MONAI-style UNet with Dynamic Active Depth
 
Remove learnable `up_bypass` and instead ALWAYS use `up_full`.
When bypassing deeper recursion, we still build the skip concat as cat(down(x), down(x)) = 2c,
but we adapt channels (2c -> upc_full) using a *frozen* 1x1 conv adapter, then feed into `up_full`.
 
Benefits:
- One shared decoder path (`up_full`) across all clients (shallow and deep).
- Shallow clients still contribute gradients to the shared `up_full` (through bypass),
  reducing "two decoder branches" mismatch in FL aggregation.
- Adapter is frozen by default so it won't become a separate learnable pathway.
 
Notes:
- For most levels, upc_full == 2c, so the adapter becomes Identity and adds no compute.
- Only at the penultimate level (where bottom outputs channels[1]) you have upc_full = c + channels[1],
  so the adapter is a real 1x1 conv (frozen).
"""
 
from __future__ import annotations
 
import math
import warnings
from typing import Sequence, Union, Optional
 
import torch
import torch.nn as nn
import torch.nn.functional as F
 
from monai.networks.blocks import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
 

class _BottomAdapter(nn.Module):
    """Adapts a plain nn.Module(bottom layer) to accept (x, active_layers)."""
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
 
    def forward(self, x: torch.Tensor, active_layers: int) -> torch.Tensor:
        return self.module(x)
 

class ConditionalUNetBlock(nn.Module):
    """
    One UNet "level" that supports dynamic depth activation.
 
    behavior:
    - There is only ONE up module: up_full
    - On bypass path, we replace subblock output with x_down (so concat becomes 2c),
      then run a frozen adapter (2c -> upc_full) if needed, then run up_full.
    """
    def __init__(
        self,
        down: nn.Module,
        subblock: nn.Module,
        up_full: nn.Module,
        bypass_adapter: nn.Module,
        level_idx: int,
    ):
        super().__init__()
        self.down = down
        self.subblock = subblock
        self.up_full = up_full
        self.bypass_adapter = bypass_adapter
        self.level_idx = level_idx
 
    def forward(self, x: torch.Tensor, active_layers: int) -> torch.Tensor:
        x_down = self.down(x)
 
        # Bypass deeper recursion: replace subblock output with x_down
        if self.level_idx >= active_layers:
            x_cat = torch.cat([x_down, x_down], dim=1)   # (N, 2c, ...)
            x_cat = self.bypass_adapter(x_cat)           # (N, upc_full, ...)
            return self.up_full(x_cat)
 
        # Normal recursion
        x_sub = self.subblock(x_down, active_layers)
        x_cat = torch.cat([x_down, x_sub], dim=1)        # (N, upc_full, ...)
        return self.up_full(x_cat)
 
    def extract_features(self, x: torch.Tensor, active_layers: int) -> dict:
        """
        Run the encoder/down path once up to `active_layers` and cache all features
        needed by every shallower depth. This is used by UNetMultiDepth to avoid
        recomputing encoder features for depths 1..K.
        """
        x_down = self.down(x)
        features = {"x_down": x_down}
 
        # Only descend if the deepest requested exit needs the child branch.
        if self.level_idx < active_layers:
            if isinstance(self.subblock, ConditionalUNetBlock):
                features["subblock"] = self.subblock.extract_features(x_down, active_layers)
            else:
                # Bottom layer output is cached too, so the deepest exit does not
                # recompute the bottom block.
                features["subblock"] = self.subblock(x_down, active_layers)
 
        return features
 
    def decode_from_features(self, features: dict, active_layers: int) -> torch.Tensor:
        """Decode one depth from cached features produced by `extract_features`."""
        x_down = features["x_down"]
 
        # Same bypass behavior as forward(), but without rerunning self.down.
        if self.level_idx >= active_layers:
            x_cat = torch.cat([x_down, x_down], dim=1)
            x_cat = self.bypass_adapter(x_cat)
            return self.up_full(x_cat)
 
        if isinstance(self.subblock, ConditionalUNetBlock):
            x_sub = self.subblock.decode_from_features(features["subblock"], active_layers)
        else:
            x_sub = features["subblock"]
 
        x_cat = torch.cat([x_down, x_sub], dim=1)
        return self.up_full(x_cat)
 
 
class UNet(nn.Module):
    """
    MONAI-style Residual UNet with dynamic "active_layers" that bypasses deeper recursion.
 
    This implementation allows for dynamic control over the depth of the network by specifying
    the number of active layers. This is useful for scenarios where you want to reduce
    computational cost by not using all the layers in the network.
 
    Behaviour:
    - Always use `up_full` for decoding.
    - Use a frozen 1x1 adapter on bypass path when (2c != upc_full).
    """
 
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[tuple, str] = Act.PRELU,
        norm: Union[tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        active_layers: Optional[int] = None,
        freeze_bypass_adapter: bool = True,  
    ) -> None:
        super().__init__()
 
        # --- Validation ---
        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError("the length of `strides` should equal to `len(channels) - 1`.")
        if delta > 0:
            warnings.warn(f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used.")
 
        if isinstance(kernel_size, Sequence) and len(kernel_size) != spatial_dims:
            raise ValueError("If sequence, the length of `kernel_size` should equal `spatial_dims`.")
        if isinstance(up_kernel_size, Sequence) and len(up_kernel_size) != spatial_dims:
            raise ValueError("If sequence, the length of `up_kernel_size` should equal `spatial_dims`.")
 
        # Store config
        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = tuple(channels)
        self.strides = tuple(strides[: len(channels) - 1])  # ignore extra
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering
        self.freeze_bypass_adapter = freeze_bypass_adapter
 
        # Active depth default: full depth
        if active_layers is None:
            active_layers = len(self.channels)
        if not (1 <= active_layers <= len(self.channels)):
            raise ValueError(f"`active_layers` must be in [1, {len(self.channels)}], got {active_layers}.")
        self.active_layers = int(active_layers)
 
        # Build full model
        self.model = self._create_block(
            inc=self.in_channels,
            outc=self.out_channels,
            channels=self.channels,
            strides=self.strides,
            is_top=True,
            level_idx=1,
        )
 
    # ----------------------------
    # Layers
    # ----------------------------
 
    def _get_down_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        if self.num_res_units > 0:
            return ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                adn_ordering=self.adn_ordering,
            )
 
        return Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )
 
    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        return self._get_down_layer(in_channels, out_channels, strides=1, is_top=False)
 
    def _get_up_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        conv: nn.Module = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
        )
 
        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_top,
                adn_ordering=self.adn_ordering,
            )
            conv = nn.Sequential(conv, ru)
 
        return conv
 
    def _make_1x1_conv(self, in_ch: int, out_ch: int) -> nn.Module:
        if in_ch == out_ch:
            return nn.Identity()
 
        if self.dimensions == 1:
            conv = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        elif self.dimensions == 2:
            conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        elif self.dimensions == 3:
            conv = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)
        else:
            raise ValueError(f"Unsupported spatial_dims={self.dimensions}")
 
        if self.freeze_bypass_adapter:
            for p in conv.parameters():
                p.requires_grad = False
        return conv
 
    # ----------------------------
    # Recursive builder
    # ----------------------------
 
    def _create_block(
        self,
        inc: int,
        outc: int,
        channels: Sequence[int],
        strides: Sequence[int],
        is_top: bool,
        level_idx: int,
    ) -> nn.Module:
        """
        Build from top to bottom recursively, returning a module that accepts (x, active_layers).
        """
        c = channels[0]
        s = strides[0]  # stride for this level down/up
 
        down = self._get_down_layer(inc, c, s, is_top=is_top)
 
        # Build subblock or bottom
        if len(channels) > 2:
            subblock = self._create_block(
                inc=c,
                outc=c,
                channels=channels[1:],
                strides=strides[1:],
                is_top=False,
                level_idx=level_idx + 1,
            )
            # Full path concat channels: cat(c, c) => 2c
            upc_full = 2 * c
        else:
            # bottom maps c -> channels[1]
            bottom = self._get_bottom_layer(c, channels[1])
            subblock = _BottomAdapter(bottom)
 
            # Full path concat channels: cat(down_out(c), bottom_out(channels[1])) => c + channels[1]
            upc_full = c + channels[1]
 
        # Single decoder module
        up_full = self._get_up_layer(upc_full, outc, s, is_top=is_top)
 
        # Bypass concat is always 2c; adapt 2c -> upc_full if needed
        bypass_adapter = self._make_1x1_conv(in_ch=2 * c, out_ch=upc_full)
 
        return ConditionalUNetBlock(
            down=down,
            subblock=subblock,
            up_full=up_full,
            bypass_adapter=bypass_adapter,
            level_idx=level_idx,
        )
 
    # ----------------------------
    # FL utils
    # ----------------------------
 
    def get_learnable_parameters(self, active_layers: Optional[int] = None):
        """
        Return parameters that SHOULD be trained given active_layers, matching execution.
 
        Behaviour:
        - down always runs
        - if bypass: bypass_adapter runs (but is typically frozen) + up_full runs
        - if full: subblock runs + up_full runs
 
        We exclude frozen bypass_adapter parameters automatically (requires_grad=False).
        """
        k = self.active_layers if active_layers is None else int(active_layers)
        if not (1 <= k <= len(self.channels)):
            raise ValueError(f"active_layers must be in [1, {len(self.channels)}], got {k}")
 
        learnable = []
 
        def add_params(prefix: str, mod: nn.Module):
            for n, p in mod.named_parameters(recurse=True):
                if p.requires_grad:
                    learnable.append((f"{prefix}.{n}", p))
 
        def collect_from_module(mod: nn.Module, prefix: str):
            if isinstance(mod, ConditionalUNetBlock):
                add_params(f"{prefix}.down", mod.down)
 
                if mod.level_idx >= k:
                    # bypass: adapter (usually frozen) + up_full
                    add_params(f"{prefix}.bypass_adapter", mod.bypass_adapter)
                    add_params(f"{prefix}.up_full", mod.up_full)
                    return
 
                # full path: recurse + up_full
                collect_from_module(mod.subblock, f"{prefix}.subblock")
                add_params(f"{prefix}.up_full", mod.up_full)
                return
 
            if isinstance(mod, _BottomAdapter):
                add_params(f"{prefix}.module", mod.module)
                return
 
            add_params(prefix, mod)
 
        collect_from_module(self.model, "model")
 
        # De-duplicate by parameter identity
        seen = set()
        unique = []
        for name, p in learnable:
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            unique.append((name, p))
 
        return unique
 
    def estimate_flops(
        self,
        input_shape,
        active_layers: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
    ) -> int:
        """
        Device-agnostic forward FLOPs estimate on CPU.
        Returns total forward FLOPs (int).
        """
        from torch.utils.flop_counter import FlopCounterMode
 
        k = self.active_layers if active_layers is None else int(active_layers)
        if not (1 <= k <= len(self.channels)):
            raise ValueError(f"active_layers must be in [1, {len(self.channels)}], got {k}")
 
        orig_device = next(self.parameters()).device
        self_cpu = self.to("cpu").eval()
 
        x = torch.randn(*input_shape, device="cpu", dtype=dtype)
 
        with FlopCounterMode(display=False) as fc:
            _ = self_cpu.forward(x, active_layers=k)
 
        if hasattr(fc, "get_total_flops"):
            total_flops = int(fc.get_total_flops())
        else:
            total_flops = int(getattr(fc, "total_flops", 0))
 
        self.to(orig_device)
        return total_flops
 
    # ----------------------------
    # Forward
    # ----------------------------
 
    def _extract_features(self, x: torch.Tensor, active_layers: Optional[int] = None) -> dict:
        k = self.active_layers if active_layers is None else int(active_layers)
        if not (1 <= k <= len(self.channels)):
            raise ValueError(f"active_layers must be in [1, {len(self.channels)}], got {k}.")
        return self.model.extract_features(x, k)
 
    def _decode_from_features(self, features: dict, active_layers: Optional[int] = None) -> torch.Tensor:
        k = self.active_layers if active_layers is None else int(active_layers)
        if not (1 <= k <= len(self.channels)):
            raise ValueError(f"active_layers must be in [1, {len(self.channels)}], got {k}.")
        return self.model.decode_from_features(features, k)
 
    def forward(self, x: torch.Tensor, active_layers: Optional[int] = None) -> torch.Tensor:
        k = self.active_layers if active_layers is None else int(active_layers)
        if not (1 <= k <= len(self.channels)):
            raise ValueError(f"active_layers must be in [1, {len(self.channels)}], got {k}.")
        return self.model(x, k)
 
 
class UNetMultiDepth(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[tuple, str] = Act.PRELU,
        norm: Union[tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        freeze_bypass_adapter: bool = True,  
        active_layers: int = 5,
    ) -> None:
        super().__init__()
 
        self.model = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            kernel_size=kernel_size,
            up_kernel_size=up_kernel_size,
            num_res_units=num_res_units,
            act=act,
            norm=norm,
            dropout=dropout,
            bias=bias,
            adn_ordering=adn_ordering,
            active_layers = active_layers,
            freeze_bypass_adapter=freeze_bypass_adapter,
 
        )
        self.active_layers = active_layers
 
    def get_learnable_parameters(self):
        seen = set()
        result = []
 
        for active_layers in range(1, self.active_layers + 1):
            params = self.model.get_learnable_parameters(
                active_layers=active_layers,
            )
 
            for name, param in params:
                if id(param) not in seen:
                    seen.add(id(param))
                    result.append((name, param))
 
        return result
 
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = self.model._extract_features(x, active_layers=self.active_layers)
        return [
            self.model._decode_from_features(features, active_layers=exit_depth)
            for exit_depth in range(1, self.active_layers + 1)
        ]
 
    def estimate_flops(
        self,
        input_shape,
        dtype: torch.dtype = torch.float32,
    ) -> int:
        """Estimate FLOPs for the cached multi-depth forward pass."""
        from torch.utils.flop_counter import FlopCounterMode
 
        orig_device = next(self.parameters()).device
        was_training = self.training
        self_cpu = self.to("cpu").eval()
        x = torch.randn(*input_shape, device="cpu", dtype=dtype)
 
        try:
            with torch.no_grad():
                with FlopCounterMode(display=False) as fc:
                    _ = self_cpu.forward(x)
 
            if hasattr(fc, "get_total_flops"):
                total_flops = int(fc.get_total_flops())
            else:
                total_flops = int(getattr(fc, "total_flops", 0))
        finally:
            self.to(orig_device)
            self.train(was_training)
 
        return total_flops
 
 
 
if __name__ == "__main__":
    # ----------------------------
    # Shared test configuration
    # ----------------------------
    compute_capacities = [1, 2, 1, 4, 5]
    input_shape = (10, 4, 160, 160, 160)
 
 
    spatial_shape = input_shape[2:]
    spatial_dims = len(spatial_shape)
    in_channels = input_shape[1]
    out_channels = 3
    channels = (16, 32, 64, 128, 256)
    strides = (2, 2, 2, 2)
    batch_size = input_shape[0]
    active_layer_values = range(1, len(channels) + 1)

    print("\nConfiguration:")
    print(f"  input_shape:        {input_shape}")
    print(f"  batch_size:         {batch_size}")
    print(f"  in_channels:        {in_channels}")
    print(f"  out_channels:       {out_channels}")
    print(f"  spatial_shape:      {spatial_shape}")
    print(f"  spatial_dims:       {spatial_dims}")
    print(f"  channels:           {channels}")
    print(f"  strides:            {strides}")
    print(f"  active_layers:      1..{len(channels)}")
 
    x = torch.randn(*input_shape)
 
    model_kwargs = {
        "spatial_dims": spatial_dims,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "channels": channels,
        "strides": strides,
        "act": "RELU",
    }
 
    # ----------------------------
    # Multi-depth UNet output shapes
    # ----------------------------
    # check if every layer is producing the expected output shape as the input shape
    multi_depth_net = UNetMultiDepth(
        **model_kwargs,
        active_layers=len(channels),
    )
 
    outputs = multi_depth_net(x)
 
    print("\nMulti-depth UNet output shapes:")
    for depth_idx, output in enumerate(outputs, start=1):
        print(f"  Depth {depth_idx}: shape={tuple(output.shape)}")
 
    # ----------------------------
    # Multi-depth UNet FLOPs
    # ----------------------------
    print("\nMulti-depth UNet FLOPs:")
    for active_layers in active_layer_values:
        multi_depth_net = UNetMultiDepth(
            **model_kwargs,
            active_layers=active_layers,
        )
 
        total_flops = multi_depth_net.estimate_flops(input_shape=input_shape)
 
        print(
            f"  active_layers={active_layers}: "
            f"{total_flops / 1e9:.3f} GFLOPs"
        )
 
    # ----------------------------
    # Single-depth UNet FLOPs
    # ----------------------------
    print("\nSingle-depth UNet FLOPs:")
    for active_layers in active_layer_values:
        single_depth_net = UNet(
            **model_kwargs,
            active_layers=active_layers,
        )
 
        flops = single_depth_net.estimate_flops(
            input_shape=input_shape,
            active_layers=active_layers,
        )
 
        print(
            f"  active_layers={active_layers}: "
            f"{flops / 1e9:.3f} GFLOPs"
        )

    # ----------------------------
    # Learnable parameter inspection
    # ----------------------------
    print("\nLearnable parameter inspection: single-depth UNet")
    for client_idx, compute_capacity in enumerate(compute_capacities, start=1):
        single_depth_net = UNet(
            **model_kwargs,
            active_layers=compute_capacity,
        )
 
        learnable_params = single_depth_net.get_learnable_parameters(
            active_layers=compute_capacity,
        )
 
        print(
            f"\nClient {client_idx}: "
            f"compute_capacity={compute_capacity}, "
            f"learnable params={len(learnable_params)}"
        )
 
        for name, param in learnable_params:
            print(f"  {name}: {tuple(param.shape)}")
 
     # ----------------------------
    # Learnable parameter inspection: multi-depth UNet
    # ----------------------------
    print("\nLearnable parameter inspection: multi-depth UNet")
 
    for active_layers in active_layer_values:
        multi_depth_net = UNetMultiDepth(
            **model_kwargs,
            active_layers=active_layers,
        )
 
        learnable_params = multi_depth_net.get_learnable_parameters()
 
        total_learnable_params = sum(
            param.numel() for _, param in learnable_params
        )
 
        print(
            f"\nMulti-depth UNet: "
            f"active_layers={active_layers}, "
            f"learnable tensors={len(learnable_params)}, "
            f"learnable parameters={total_learnable_params:,}"
        )
 
        for name, param in learnable_params:
            print(
                f"  {name}: "
                f"shape={tuple(param.shape)}, "
                f"numel={param.numel():,}"
            )
 