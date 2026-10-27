"""Fixed 128/256/512 ROI refinement over frozen SAM2 features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from utils.geometry import _crop_grid, _inverse_grid
from types import SimpleNamespace

if TYPE_CHECKING:
    from model.sam2.modeling.sam.mask_decoder import SingleMaskDecoderFeatures


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        *,
        dilation: int = 1,
    ) -> None:
        padding = dilation * (kernel_size // 2)
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        )


class MultiScaleContext(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            ConvNormAct(channels, channels, dilation=dilation) for dilation in (1, 3, 5)
        )
        self.merge = ConvNormAct(channels, channels, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        merged = F.adaptive_avg_pool2d(inputs, 1)
        for branch in self.branches:
            merged = merged + branch(inputs)
        return self.merge(merged)


class RefinementStage(nn.Module):
    """Predict one signed residual conditioned on the current logits."""

    def __init__(self, in_channels: int, width: int) -> None:
        super().__init__()
        logit_width = max(width // 4, 8)
        self.logit_encoder = ConvNormAct(1, logit_width)
        self.input_projection = ConvNormAct(
            in_channels + logit_width, width, kernel_size=1
        )
        self.context = MultiScaleContext(width)
        self.output_projection = ConvNormAct(width, width, kernel_size=1)
        hidden = max(width // 2, 8)
        self.residual_head = nn.Sequential(
            ConvNormAct(width, hidden),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def forward(
        self,
        inputs: torch.Tensor,
        current_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logit_features = self.logit_encoder(current_logits.float())
        inputs = torch.cat((inputs, logit_features), dim=1)
        state = self.input_projection(inputs)
        state = self.output_projection(state + self.context(state))
        return state, self.residual_head(state)


@dataclass(frozen=True)
class SpatialRefineOutput:
    logits: torch.Tensor
    base_logits: torch.Tensor
    residual: torch.Tensor
    roi_boxes: torch.Tensor
    stage_sizes: tuple[int, ...]


def expand_boxes(
    boxes: torch.Tensor,
    image_size: tuple[int, int] | torch.Size,
    context_scale: float = 1.25,
) -> torch.Tensor:
    height, width = map(int, image_size)
    x0, y0, x1, y1 = boxes.float().unbind(dim=1)
    center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
    box_width = (x1 - x0).clamp_min(1.0) * context_scale
    box_height = (y1 - y0).clamp_min(1.0) * context_scale
    left = (center_x - box_width / 2).clamp(0.0, float(width - 1))
    right = (center_x + box_width / 2).clamp(0.0, float(width - 1))
    top = (center_y - box_height / 2).clamp(0.0, float(height - 1))
    bottom = (center_y + box_height / 2).clamp(0.0, float(height - 1))
    right = torch.maximum(right, (left + 1).clamp_max(float(width - 1)))
    bottom = torch.maximum(bottom, (top + 1).clamp_max(float(height - 1)))
    left = torch.minimum(left, right - 1)
    top = torch.minimum(top, bottom - 1)
    return torch.stack((left, top, right, bottom), dim=1)


class ReferenceHead(nn.Module):
    """Semantic-to-structure-to-detail refinement with fixed 128/256/512 ROI sizes."""

    def __init__(
        self,
        *,
        width: int = 64,
        sam_s0_channels: int = 32,
        sam_s1_channels: int = 64,
        sam_embedding_channels: int = 256,
        max_logit_residual: float = 8.0,
    ) -> None:
        super().__init__()

        self.width = int(width)
        self.max_logit_residual = float(max_logit_residual)
        self.stage_sizes = (128, 256, 512)
        detail_width = width

        self.sam16_projection = ConvNormAct(
            sam_embedding_channels, width, kernel_size=1
        )
        self.sam8_projection = ConvNormAct(sam_s1_channels, width, kernel_size=1)
        self.sam4_projection = ConvNormAct(sam_s0_channels, width, kernel_size=1)
        self.structure_fusion = ConvNormAct(width * 2, width, kernel_size=1)
        self.semantic_seed = nn.Sequential(
            ConvNormAct(width * 2 + detail_width, width, kernel_size=1),
            MultiScaleContext(width),
        )
        self.stages = nn.ModuleList(
            (
                RefinementStage(width, width),
                RefinementStage(width * 2 + detail_width, width),
                RefinementStage(width * 2 + detail_width, width),
            )
        )
        # Match the original parameter order for gradient clipping and optimizer state.
        self.encoder_detail = nn.Sequential(ConvNormAct(144, width, kernel_size=1))

    @staticmethod
    def _sample(
        tensor: torch.Tensor,
        boxes: torch.Tensor,
        size: int,
        image_height: int,
        image_width: int,
        *,
        mode: str = "bilinear",
    ) -> torch.Tensor:
        grid = _crop_grid(boxes.float(), size, image_height, image_width)
        return F.grid_sample(
            tensor,
            grid.to(tensor.dtype),
            mode=mode,
            padding_mode="border",
            align_corners=True,
        )

    @staticmethod
    def _resize(tensor: torch.Tensor, size: int) -> torch.Tensor:
        kwargs = {"antialias": True} if tensor.shape[-1] > size else {}
        return F.interpolate(
            tensor.float(),
            size=(size, size),
            mode="bilinear",
            align_corners=False,
            **kwargs,
        )

    def _guidance(
        self,
        features: SingleMaskDecoderFeatures,
        roi_boxes: torch.Tensor,
        sizes: tuple[int, int, int],
        image_height: int,
        image_width: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        semantic_size, structure_size, detail_size = sizes
        semantic = torch.cat(
            (
                self.sam16_projection(
                    self._sample(
                        features.embedding_16,
                        roi_boxes,
                        semantic_size,
                        image_height,
                        image_width,
                    )
                ),
                self.sam8_projection(
                    self._sample(
                        features.embedding_8,
                        roi_boxes,
                        semantic_size,
                        image_height,
                        image_width,
                    )
                ),
            ),
            dim=1,
        )
        structure = self.structure_fusion(
            torch.cat(
                (
                    self.sam8_projection(
                        self._sample(
                            features.embedding_8,
                            roi_boxes,
                            structure_size,
                            image_height,
                            image_width,
                        )
                    ),
                    self.sam4_projection(
                        self._sample(
                            features.embedding_4,
                            roi_boxes,
                            structure_size,
                            image_height,
                            image_width,
                        )
                    ),
                ),
                dim=1,
            )
        )
        detail = self.sam4_projection(
            self._sample(
                features.embedding_4,
                roi_boxes,
                detail_size,
                image_height,
                image_width,
            )
        )
        return semantic, structure, detail

    def _detail_features(
        self, decoder_features, roi_boxes, sizes, image_height, image_width
    ):
        crop = self._sample(
            decoder_features.encoder_detail,
            roi_boxes,
            sizes[-1],
            image_height,
            image_width,
        )
        detail = self.encoder_detail(crop)
        return self._resize(detail, sizes[0]), self._resize(detail, sizes[1]), detail

    @torch.no_grad()
    def extract_frozen_inputs(self, model, images, boxes):
        captured = []

        def capture(module, inputs, output):
            captured.append(output.detach().permute(0, 3, 1, 2).contiguous())

        handle = model.image_encoder.trunk.blocks[1].register_forward_hook(capture)
        try:
            coarse, token, decoder = model.forward_single_mask_with_decoder_features(
                images, boxes=boxes, encoder_trainable=False
            )
        finally:
            handle.remove()
        return (
            coarse,
            token,
            SimpleNamespace(**vars(decoder), encoder_detail=captured[0]),
        )

    def forward(
        self,
        coarse_logits: torch.Tensor,
        roi_boxes: torch.Tensor,
        output_size: tuple[int, int] | torch.Size,
        *,
        decoder_features: SingleMaskDecoderFeatures,
    ) -> SpatialRefineOutput:
        image_height, image_width = map(int, output_size)

        sizes = self.stage_sizes
        base_logits = F.interpolate(
            coarse_logits.float(),
            size=(image_height, image_width),
            mode="bilinear",
            align_corners=False,
        )
        detail_features = self._detail_features(
            decoder_features, roi_boxes, sizes, image_height, image_width
        )
        guidance = self._guidance(
            decoder_features,
            roi_boxes,
            sizes,
            image_height,
            image_width,
        )

        cumulative: torch.Tensor | None = None
        state: torch.Tensor | None = None
        for index, size in enumerate(sizes):
            stage_base = self._sample(
                base_logits, roi_boxes, size, image_height, image_width
            )
            if index == 0:
                stage_inputs = self.semantic_seed(
                    torch.cat((guidance[index], detail_features[index]), dim=1)
                )
                cumulative = torch.zeros_like(stage_base)
            else:
                state = self._resize(state, size)
                cumulative = self._resize(cumulative, size)
                stage_inputs = torch.cat(
                    (state, guidance[index], detail_features[index]), dim=1
                )
            current_logits = stage_base + cumulative
            state, raw_delta = self.stages[index](stage_inputs, current_logits)
            delta = self.max_logit_residual * torch.tanh(raw_delta)
            cumulative = cumulative + delta

        inverse_grid, valid = _inverse_grid(
            roi_boxes.float(), image_height, image_width
        )
        residual = F.grid_sample(
            cumulative,
            inverse_grid.to(cumulative.dtype),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        residual = residual * valid.to(residual.dtype)
        output = SpatialRefineOutput(
            logits=base_logits + residual.float(),
            base_logits=base_logits,
            residual=residual.float(),
            roi_boxes=roi_boxes.float(),
            stage_sizes=sizes,
        )
        return output


class Block1Head(ReferenceHead):
    """Full evaluation path; arithmetic order matches the saved paper evaluation."""
    def __init__(self, *, width=64, max_logit_residual=8.0):
        super().__init__(width=width, max_logit_residual=max_logit_residual)
        self.active_stages = (0, 1, 2)
        self.use_block1 = True
        self.transfer_state = True
        self.stages = nn.ModuleDict({str(i): stage for i, stage in enumerate(self.stages)})

    def forward(self, coarse_logits, roi_boxes, output_size, *, decoder_features):
        height, width = map(int, output_size)
        base = F.interpolate(
            coarse_logits.float(),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        features = decoder_features
        if self.use_block1:
            detail512 = self.encoder_detail(
                self._sample(features.encoder_detail, roi_boxes, 512, height, width)
            )
        cumulative, state = None, None
        for position, index in enumerate(self.active_stages):
            size = (128, 256, 512)[index]

            def crop(value):
                return self._sample(value, roi_boxes, size, height, width)

            if index == 0:
                guidance = torch.cat(
                    (
                        self.sam16_projection(crop(features.embedding_16)),
                        self.sam8_projection(crop(features.embedding_8)),
                    ),
                    dim=1,
                )
            elif index == 1:
                guidance = self.structure_fusion(
                    torch.cat(
                        (
                            self.sam8_projection(crop(features.embedding_8)),
                            self.sam4_projection(crop(features.embedding_4)),
                        ),
                        dim=1,
                    )
                )
            else:
                guidance = self.sam4_projection(crop(features.embedding_4))
            parts = []
            if position > 0 and self.transfer_state:
                parts.append(self._resize(state, size))
            parts.append(guidance)
            if self.use_block1:
                parts.append(
                    detail512 if size == 512 else self._resize(detail512, size)
                )
            inputs = torch.cat(parts, dim=1)
            if index == 0:
                inputs = self.semantic_seed(inputs)
            stage_base = crop(base)
            cumulative = (
                torch.zeros_like(stage_base)
                if cumulative is None
                else self._resize(cumulative, size)
            )
            current = stage_base + cumulative
            state, delta = self.stages[str(index)](inputs, current)
            cumulative = cumulative + self.max_logit_residual * torch.tanh(delta)
        grid, valid = _inverse_grid(roi_boxes.float(), height, width)
        residual = F.grid_sample(
            cumulative,
            grid.to(cumulative.dtype),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        residual = residual * valid.to(residual.dtype)
        return SpatialRefineOutput(
            base + residual.float(),
            base,
            residual.float(),
            roi_boxes.float(),
            self.stage_sizes,
        )
