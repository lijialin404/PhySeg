"""Independent Q/V low-rank updates; Key and the fused base weights stay frozen."""

import math
import torch
from torch import nn
from model.sam2.build_sam import build_sam2


class QueryValueLoRALinear(nn.Module):
    """Frozen fused QKV linear with independent Q/V LoRA."""

    factor_names = ("lora_a_q", "lora_b_q", "lora_a_v", "lora_b_v")

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
        layer_name: str,
    ) -> None:
        super().__init__()

        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.qkv_features = base.out_features // 3
        self.layer_name = str(layer_name)

        kwargs = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_a_q = nn.Linear(base.in_features, self.rank, bias=False, **kwargs)
        self.lora_b_q = nn.Linear(self.rank, self.qkv_features, bias=False, **kwargs)
        self.lora_a_v = nn.Linear(base.in_features, self.rank, bias=False, **kwargs)
        self.lora_b_v = nn.Linear(self.rank, self.qkv_features, bias=False, **kwargs)
        nn.init.kaiming_uniform_(self.lora_a_q.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b_q.weight)
        nn.init.kaiming_uniform_(self.lora_a_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b_v.weight)
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        query, key, value = self.base(inputs).split(self.qkv_features, dim=-1)
        dropped = self.dropout(inputs)
        query = query + self.lora_b_q(self.lora_a_q(dropped)) * self.scaling
        value = value + self.lora_b_v(self.lora_a_v(dropped)) * self.scaling
        return torch.cat((query, key, value), dim=-1)


def build_model(cfg, device="cuda"):
    model = build_sam2(
        cfg.checkpoint,
        device=device,
        image_size=cfg.image_size,
    )
    for index, block in enumerate(model.image_encoder.trunk.blocks):
        block.attn.qkv = QueryValueLoRALinear(
            block.attn.qkv,
            rank=cfg.rank,
            alpha=cfg.alpha,
            dropout=cfg.dropout,
            layer_name=f"blocks.{index}.attn.qkv",
        )
    model.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, QueryValueLoRALinear):
            for factor in module.factor_names:
                getattr(module, factor).weight.requires_grad_(True)
    set_pure_lora_mode(model, False)
    return model


def set_pure_lora_mode(model, training):
    # Only LoRA dropout changes mode. The pretrained backbone/decoder remain in eval.
    model.eval()
    for module in model.modules():
        if isinstance(module, QueryValueLoRALinear):
            module.dropout.train(training)


def lora_state_dict(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in model.named_parameters()
        if name.endswith(
            tuple(f".{factor}.weight" for factor in QueryValueLoRALinear.factor_names)
        )
    }


def load_lora_state(model, state):
    suffixes = tuple(
        f".{factor}.weight" for factor in QueryValueLoRALinear.factor_names
    )
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith(suffixes):
                parameter.copy_(state[name])
