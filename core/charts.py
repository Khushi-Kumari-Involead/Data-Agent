import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import uuid
import os
from typing import Tuple

def make_chart_from_df(df: pd.DataFrame, chart_type: str, x: str, y: str) -> Tuple[plt.Figure, str]:
    """Generate a chart from a DataFrame and save it to disk."""
    fig, ax = plt.subplots(figsize=(7, 4))
    try:
        df[x] = pd.to_numeric(df[x], errors="coerce")
        df[y] = pd.to_numeric(df[y], errors="coerce")
        df = df.dropna(subset=[x, y])
    except Exception:
        pass
    if df.empty:
        raise ValueError("No numeric data to plot after coercion.")
    if chart_type == "bar":
        ax.bar(df[x].astype(str), df[y])
    elif chart_type == "line":
        ax.plot(df[x], df[y], marker="o")
    elif chart_type == "scatter":
        ax.scatter(df[x], df[y])
    else:
        raise ValueError("Unsupported chart type.")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{y} vs {x}")
    fig.tight_layout()
    fname = os.path.join("charts", f"{uuid.uuid4().hex}.png")
    fig.savefig(fname)
    return fig, fname