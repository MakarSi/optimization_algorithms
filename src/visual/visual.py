from typing import Callable, List, Tuple
import plotly.graph_objects as go
import numpy as np


def plot_3dpath(
    func: Callable[[float, float], float],
    path: List[Tuple[float, float]],
    x_range: float,
    y_range: float,
    title: str,
):
    x_min, x_max = x_range
    y_min, y_max = y_range
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    Z = np.array([[func(xi, yi) for xi in x] for yi in y])

    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]
    path_z = [func(p[0], p[1]) for p in path]

    fig = go.Figure(
        data=[
            go.Surface(x=x, y=y, z=Z, colorscale="Viridis", opacity=0.7),
            go.Scatter3d(
                x=path_x,
                y=path_y,
                z=path_z,
                mode="markers+lines",
                marker=dict(size=5, color="red"),
                line=dict(width=1, color="red"),
                name="Шаги оптимизации",
            ),
        ]
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="f(x, y)",
        ),
        title=title,
        margin=dict(l=20, r=20, b=20, t=50),
    )
    fig.write_html(f"plots/{title}.html")
