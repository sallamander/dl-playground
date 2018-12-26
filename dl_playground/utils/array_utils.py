

def get_padding_size(shape1, shape2):
    """Get padding necessary to pad shape1 to shape2"""

    padding = [
        dim_val2 - dim_val1 for dim_val1, dim_val2 in zip(shape1, shape2)
    ]
    return padding
