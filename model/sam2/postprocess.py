import torch.nn.functional as F


def upsample_mask_logits(logits, output_size):
    return F.interpolate(
        logits.float(), size=tuple(output_size), mode="bilinear", align_corners=False
    )


def masks_from_logits(logits, output_size, *, threshold=0.0):
    return upsample_mask_logits(logits, output_size) > threshold
