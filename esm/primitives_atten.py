import math
import torch
import torch.nn as nn
from typing import Optional, List, Tuple

class Attention(nn.Module):
    """
    Standard multi-head attention using only native PyTorch.
    Compatible with original state dict.
    """
    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        no_heads: int,
        gating: bool = True,
    ):
        """
        Args:
            c_q:
                Input dimension of query data
            c_k:
                Input dimension of key data
            c_v:
                Input dimension of value data
            c_hidden:
                Per-head hidden dimension
            no_heads:
                Number of attention heads
            gating:
                Whether the output should be gated using query data
        """
        super(Attention, self).__init__()

        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating

        # DISCREPANCY: c_hidden is not the per-head channel dimension, as
        # stated in the supplement, but the overall channel dimension.

        # Use nn.Linear directly with the exact same names for state dict compatibility
        self.linear_q = nn.Linear(
            self.c_q, self.c_hidden * self.no_heads, bias=False
        )
        self.linear_k = nn.Linear(
            self.c_k, self.c_hidden * self.no_heads, bias=False
        )
        self.linear_v = nn.Linear(
            self.c_v, self.c_hidden * self.no_heads, bias=False
        )
        self.linear_o = nn.Linear(
            self.c_hidden * self.no_heads, self.c_q
        )

        self.linear_g = None
        if self.gating:
            self.linear_g = nn.Linear(
                self.c_q, self.c_hidden * self.no_heads
            )

        self.sigmoid = nn.Sigmoid()

    def _prep_qkv(self,
        q_x: torch.Tensor, 
        kv_x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # [*, Q/K/V, H * C_hidden]
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        # [*, Q/K, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        # [*, H, Q/K, C_hidden]
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)

        q /= math.sqrt(self.c_hidden)

        return q, k, v

    def _wrap_up(self,
        o: torch.Tensor, 
        q_x: torch.Tensor
    ) -> torch.Tensor:
        if self.linear_g is not None:
            g = self.sigmoid(self.linear_g(q_x))
        
            # [*, Q, H, C_hidden]
            g = g.view(g.shape[:-1] + (self.no_heads, -1))
            o = o * g

        # [*, Q, H * C_hidden]
        o = o.reshape(o.shape[:-2] + (-1,))

        # [*, Q, C_q]
        o = self.linear_o(o)

        return o

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        biases: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            q_x:
                [*, Q, C_q] query data
            kv_x:
                [*, K, C_k] key data
            biases:
                List of biases that broadcast to [*, H, Q, K]
        Returns
            [*, Q, C_q] attention update
        """
        if biases is None:
            biases = []
        
        # [*, H, Q/K, C_hidden]
        q, k, v = self._prep_qkv(q_x, kv_x)

        # Compute attention scores
        # [*, H, Q, K]
        scores = torch.matmul(q, k.transpose(-1, -2))
        
        # Add biases if provided
        for bias in biases:
            scores = scores + bias
        
        # Apply softmax
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        # [*, H, Q, C_hidden]
        o = torch.matmul(attention_weights, v)
        
        # [*, Q, H, C_hidden]
        o = o.transpose(-2, -3)

        o = self._wrap_up(o, q_x)

        return o

