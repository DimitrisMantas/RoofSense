def get_fg_color(patch_color: tuple[float, float, float, float]) -> str:
    r, g, b, a = patch_color
    r, g, b = (
        c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)
    )

    # Get the relative luminance
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b

    return ".1" if y > 0.4 else "white"
