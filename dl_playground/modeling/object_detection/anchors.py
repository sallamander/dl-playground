def tile_anchor_boxes(grid_height, grid_width, scales, aspect_ratios,
                      base_anchor_size, anchor_stride):
    """Create a tiled set of anchor boxes strided along a grid in image space
