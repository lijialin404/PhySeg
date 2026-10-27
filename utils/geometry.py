"""ROI sampling grids in the original image coordinate system."""

import torch


def _crop_grid(
    rois: torch.Tensor,
    output_size: int,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    unit = torch.linspace(0.0, 1.0, output_size, device=rois.device)
    x0, y0, x1, y1 = rois.unbind(dim=1)
    xs = x0[:, None] + (x1 - x0)[:, None] * unit[None]
    ys = y0[:, None] + (y1 - y0)[:, None] * unit[None]
    xs = 2.0 * xs / max(image_width - 1, 1) - 1.0
    ys = 2.0 * ys / max(image_height - 1, 1) - 1.0
    return torch.stack(
        (
            xs[:, None, :].expand(-1, output_size, -1),
            ys[:, :, None].expand(-1, -1, output_size),
        ),
        dim=-1,
    )


def _inverse_grid(
    rois: torch.Tensor,
    image_height: int,
    image_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    x0, y0, x1, y1 = rois.unbind(dim=1)
    xs = torch.arange(image_width, device=rois.device, dtype=rois.dtype)
    ys = torch.arange(image_height, device=rois.device, dtype=rois.dtype)
    local_x = 2.0 * (xs[None] - x0[:, None]) / (x1 - x0)[:, None] - 1.0
    local_y = 2.0 * (ys[None] - y0[:, None]) / (y1 - y0)[:, None] - 1.0
    grid = torch.stack(
        (
            local_x[:, None, :].expand(-1, image_height, -1),
            local_y[:, :, None].expand(-1, -1, image_width),
        ),
        dim=-1,
    )
    valid = (grid.abs() <= 1.0).all(dim=-1, keepdim=True).permute(0, 3, 1, 2)
    return grid, valid
