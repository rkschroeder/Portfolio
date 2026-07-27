"""Shared chart colors (validated categorical order, sequential ramp, status colors)."""

CATEGORICAL = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]

SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95"]

STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"

PLOTLY_LAYOUT_DEFAULTS = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": MUTED_INK},
    "xaxis": {"gridcolor": GRIDLINE, "zerolinecolor": GRIDLINE},
    "yaxis": {"gridcolor": GRIDLINE, "zerolinecolor": GRIDLINE},
}