"""SAM2.1 image encoder and box-prompted mask decoder."""

import contextlib
import torch
from torch import nn
from torch.nn.init import trunc_normal_
from model.sam2.modeling.backbones.hieradet import Hiera
from model.sam2.modeling.backbones.image_encoder import FpnNeck, ImageEncoder
from model.sam2.modeling.position_encoding import PositionEmbeddingSine
from model.sam2.modeling.sam.mask_decoder import MaskDecoder, SingleMaskDecoderFeatures
from model.sam2.modeling.sam.prompt_encoder import BoxPromptEncoder
from model.sam2.modeling.sam.transformer import TwoWayTransformer


class SAM2BoxModel(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        stages: tuple[int, ...],
        global_att_blocks: tuple[int, ...],
        window_pos_embed_bkg_spatial_size: tuple[int, int],
        window_spec: tuple[int, ...],
        backbone_channel_list: tuple[int, ...],
        image_size: int,
    ) -> None:
        super().__init__()
        trunk = Hiera(
            embed_dim=embed_dim,
            num_heads=num_heads,
            stages=stages,
            global_att_blocks=global_att_blocks,
            window_pos_embed_bkg_spatial_size=window_pos_embed_bkg_spatial_size,
            window_spec=window_spec,
        )
        neck = FpnNeck(
            position_encoding=PositionEmbeddingSine(
                num_pos_feats=256,
                normalize=True,
                scale=None,
                temperature=10000,
                warmup_cache=False,
            ),
            d_model=256,
            backbone_channel_list=list(backbone_channel_list),
            fpn_top_down_levels=[2, 3],
            fpn_interp_model="nearest",
        )
        self.image_encoder = ImageEncoder(trunk=trunk, neck=neck, scalp=1)
        self.image_size = int(image_size)
        self.num_feature_levels = 3
        self.no_mem_embed = nn.Parameter(torch.zeros(1, 1, 256))
        trunc_normal_(self.no_mem_embed, std=0.02)

        embedding_size = image_size // 16
        self.sam_prompt_encoder = BoxPromptEncoder(
            embed_dim=256,
            image_embedding_size=(embedding_size, embedding_size),
            input_image_size=(image_size, image_size),
        )
        self.sam_mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=True,
            iou_prediction_use_sigmoid=True,
            pred_obj_scores=True,
            pred_obj_scores_mlp=True,
            use_multimask_token_for_obj_ptr=True,
            dynamic_multimask_via_stability=False,
        )

    def encode_image(self, images, *, trainable=False):
        context = contextlib.nullcontext() if trainable else torch.no_grad()
        with context:
            backbone = self.image_encoder(images)
        features = list(backbone["backbone_fpn"][-3:])
        features[0] = self.sam_mask_decoder.conv_s0(features[0])
        features[1] = self.sam_mask_decoder.conv_s1(features[1])
        no_memory = self.no_mem_embed.permute(1, 2, 0).unsqueeze(-1)
        return features[-1] + no_memory, features[:-1]

    def forward_logits(
        self,
        images: torch.Tensor,
        *,
        boxes: torch.Tensor,
        encoder_trainable: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_embedding, high_resolution = self.encode_image(
            images,
            trainable=encoder_trainable,
        )
        sparse, dense = self.sam_prompt_encoder(boxes)
        logits, predicted_iou, _, _ = self.sam_mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_resolution,
        )
        return logits.float(), predicted_iou.float()

    def forward_single_mask_with_decoder_features(
        self,
        images: torch.Tensor,
        *,
        boxes: torch.Tensor,
        encoder_trainable: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, SingleMaskDecoderFeatures]:
        """Expose the frozen target-conditioned /16, /8, and /4 pyramid."""
        image_embedding, high_resolution = self.encode_image(
            images,
            trainable=encoder_trainable,
        )
        sparse, dense = self.sam_prompt_encoder(boxes)
        logits, mask_token_0, features = (
            self.sam_mask_decoder.predict_single_mask_with_decoder_features(
                image_embeddings=image_embedding,
                image_pe=self.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                repeat_image=False,
                high_res_features=high_resolution,
            )
        )
        return logits.float(), mask_token_0.float(), features
