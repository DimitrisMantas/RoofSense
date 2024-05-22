from collections.abc import Sequence


def get_fg_color(bg_color: Sequence[float, float, float, float]) -> tuple[float, float, float, float]:
    lum = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
    return (0, 0, 0, 1) if lum > 0.5 else (1, 1, 1, 1)