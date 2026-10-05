"""Position semantics for finance-only patch tokens."""


def shared_patch_positions(position_table, token_count, patches_per_stock):
    """Repeat temporal patch positions for each stock, without a stock-ID position."""
    if patches_per_stock is None or patches_per_stock <= 0 or token_count % patches_per_stock:
        raise ValueError('Patch-only positions require complete per-stock patches')
    return position_table[:, :patches_per_stock].repeat(1, token_count // patches_per_stock, 1)
