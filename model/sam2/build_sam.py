"""Load the image components of an official SAM2.1 Large checkpoint."""

import torch
from model.sam2.image_model import SAM2BoxModel


def build_sam2(checkpoint_path, *, device="cuda", image_size=1024):
    model = SAM2BoxModel(
        embed_dim=144,
        num_heads=2,
        stages=(2, 6, 36, 4),
        global_att_blocks=(23, 33, 43),
        window_pos_embed_bkg_spatial_size=(7, 7),
        window_spec=(8, 4, 16, 8),
        backbone_channel_list=(1152, 576, 288, 144),
        image_size=image_size,
    )
    payload = checkpoint_path if isinstance(checkpoint_path, dict) else torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    source = payload.get("sam2_state", payload.get("model", payload))
    model.load_state_dict({name: source[name] for name in model.state_dict()})
    return model.to(device)
