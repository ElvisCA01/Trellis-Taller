import torch
import torch.nn as nn


class LayerNorm32(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        needed_dtype = self.weight.dtype if self.weight is not None else x.dtype # to make it work both with float16 and float32
        return super().forward(x.to(needed_dtype)).type(x.dtype)
    

class GroupNorm32(nn.GroupNorm):
    """
    A GroupNorm layer that converts to float16 before the forward pass.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        needed_dtype = self.weight.dtype if self.weight is not None else x.dtype # to make it work both with float16 and float32
        return super().forward(x.to(needed_dtype)).type(x.dtype)
    
    
class ChannelLayerNorm32(LayerNorm32):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        DIM = x.dim()
        x = x.permute(0, *range(2, DIM), 1).contiguous()
        x = super().forward(x)
        x = x.permute(0, DIM-1, *range(1, DIM-1)).contiguous()
        return x
    