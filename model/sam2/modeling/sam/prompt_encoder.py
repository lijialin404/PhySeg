# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""The box-only subset of SAM2's prompt encoder."""

from __future__ import annotations


import torch
from torch import nn

from model.sam2.modeling.position_encoding import PositionEmbeddingRandom
from model.sam2.modeling.sam2_utils import LayerNorm2d


class BoxPromptEncoder(nn.Module):
    "Encode bounding boxes with pretrained corner embeddings."

    def __init__(
        self,
        *,
        embed_dim: int,
        image_embedding_size: tuple[int, int],
        input_image_size: tuple[int, int],
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.image_embedding_size = image_embedding_size
        self.input_image_size = input_image_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        # Preserve official indices 2/3 without retaining unused point-prompt
        # parameters at indices 0/1.
        self.point_embeddings = nn.ModuleList(
            (
                nn.Identity(),
                nn.Identity(),
                nn.Embedding(1, embed_dim),
                nn.Embedding(1, embed_dim),
            )
        )
        mask_in_chans = 16
        self.mask_input_size = (
            4 * image_embedding_size[0],
            4 * image_embedding_size[1],
        )
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            nn.GELU(),
            nn.Conv2d(
                mask_in_chans // 4,
                mask_in_chans,
                kernel_size=2,
                stride=2,
            ),
            LayerNorm2d(mask_in_chans),
            nn.GELU(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def forward(self, boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError("boxes must have shape [B,4]")
        coordinates = (boxes + 0.5).reshape(-1, 2, 2)
        sparse = self.pe_layer.forward_with_coords(
            coordinates,
            self.input_image_size,
        )
        sparse[:, 0, :] += self.point_embeddings[2].weight
        sparse[:, 1, :] += self.point_embeddings[3].weight
        dense = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            boxes.shape[0],
            -1,
            self.image_embedding_size[0],
            self.image_embedding_size[1],
        )
        return sparse, dense
