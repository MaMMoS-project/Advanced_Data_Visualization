# -*- coding: utf-8 -*-
"""
Functions to read XRD data from HDF5 files

@author: williamrigaut
"""
import plotly.express as px
import numpy as np
import xarray as xr
import pandas as pd


def colorbar_layout(z_min, z_max, precision=0, title=""):
    """
    Generates a standardized colorbar.

    Parameters:
        z_min : minimum value on the colorbar
        z_max : maximum value on the colorbar
        precision: number of digits on the colorbar scale
        title (str): The title of the plot.

    Returns:
        colorbar (dict): dictionary of colorbar parameters that can be passed to a figure
    """
    z_mid = (z_min + z_max) / 2
    colorbar = dict(
        colorbar_title=dict(text=f"{title} <br>&nbsp;<br>", font=dict(size=24)),
        colorbar_tickfont=dict(size=24),
        colorbar_tickvals=[
            z_min,
            (z_min + z_mid) / 2,
            z_mid,
            (z_max + z_mid) / 2,
            z_max,
        ],  # Tick values
        colorbar_ticktext=[
            f"{z_min:.{precision}f}",
            f"{(z_min + z_mid) / 2:.{precision}f}",
            f"{z_mid:.{precision}f}",
            f"{(z_max + z_mid) / 2:.{precision}f}",
            f"{z_max:.{precision}f}",
        ],  # Tick text
    )
    return colorbar


def plot_1d(
    x,
    y,
    x_range=None,
    y_range=None,
    width=800,
    height=600,
    graph_title=None,
    xaxis_title=None,
    yaxis_title=None,
    axis_tick_font_size=22,
    axis_title_font_size=20,
):
    fig = px.line(x=x, y=y, width=width, height=height)
    fig.update_layout(
        title=graph_title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        xaxis=dict(
            tickfont=dict(size=axis_tick_font_size),
            title_font=dict(size=axis_title_font_size),
        ),
        yaxis=dict(
            tickfont=dict(size=axis_tick_font_size),
            title_font=dict(size=axis_title_font_size),
        ),
    )

    if x_range is not None:
        fig.update_xaxes(range=x_range)
    if y_range is not None:
        fig.update_yaxes(range=y_range)

    return fig


def plot_heatmap(
    data,
    graph_title=None,
    coloraxe_title=None,
    colorscale="plasma",
    range_color=(0, 100),
    width=700,
    height=600,
    axis_tick_font_size=22,
    axis_title_font_size=20,
    colorbar_tickfont_size=22,
    colorbar_title_font_size=20,
    precision=1,
):

    fig = px.imshow(
        data,
        color_continuous_scale=colorscale,
        range_color=range_color,
        aspect="equal",
        # text_auto=True,
    )  # using imshow from plotly
    fig.update_layout(
        title=graph_title,
        xaxis_title="X (mm)",
        yaxis_title="Y (mm)",
        width=width,
        height=height,
    )  # Adding title, labels and figure
    fig.update_xaxes(range=[-42.5, 42.5], autorange=False)
    fig.update_yaxes(range=[-42.5, 42.5], autorange=False)

    fig.update_layout(
        xaxis=dict(
            tickfont=dict(size=axis_tick_font_size),
            title_font=dict(size=axis_title_font_size),
            tickmode="array",
            tickvals=[-40, -20, 0, 20, 40],
        ),
        yaxis=dict(
            tickfont=dict(size=axis_tick_font_size),
            title_font=dict(size=axis_title_font_size),
            tickmode="array",
            tickvals=[-40, -20, 0, 20, 40],
        ),
    )
    fig.update_coloraxes(
        colorbar_layout(
            range_color[0], range_color[1], precision=precision, title=coloraxe_title
        )
    )
    fig.update_coloraxes(
        colorbar_tickfont_size=colorbar_tickfont_size,
        colorbar_title_font_size=colorbar_title_font_size,
    )

    return fig


def plot_scatter(
    x,
    y,
    color,
    range_color=None,
    precision=0,
    mask=None,
    x_label="X",
    y_label="Y",
    color_label="data",
    x_range=None,
    y_range=None,
    graph_title="",
    colorscale="plasma",
    width=1050,
    height=700,
    colorbar_tickfont_size=20,
    axis_tick_font_size=22,
    axis_title_font_size=20,
    marker_size=10,
):
    # Convert xarray to 1D numpy array
    if isinstance(x, xr.core.dataarray.DataArray):
        x = x.values.reshape(-1)
    if isinstance(y, xr.core.dataarray.DataArray):
        y = y.values.reshape(-1)
    if isinstance(color, xr.core.dataarray.DataArray):
        color = color.values.reshape(-1)

    # Convert numpy 2D array to 1D array
    if isinstance(x, np.ndarray):
        if x.ndim > 1:
            x = x.reshape(-1)
    if isinstance(y, np.ndarray):
        if y.ndim > 1:
            y = y.reshape(-1)
    if isinstance(color, np.ndarray):
        if color.ndim > 1:
            color = color.reshape(-1)

    df = pd.DataFrame(
        {
            f"{x_label}": x,
            f"{y_label}": y,
            f"{color_label}": color,
        }
    )

    if mask is not None:
        mask = (df[f"{color_label}"] >= mask[0]) & (df[f"{color_label}"] <= mask[1])
        df = df[mask]

    if range_color is None:
        range_color = (df[f"{color_label}"].min(), df[f"{color_label}"].max())

    fig = px.scatter(
        df,
        x=f"{x_label}",
        y=f"{y_label}",
        color=f"{color_label}",
        range_color=range_color,
    )
    fig.update_layout(
        title=graph_title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        xaxis=dict(
            tickfont=dict(size=axis_tick_font_size),
            title_font=dict(size=axis_title_font_size),
        ),
        yaxis=dict(
            tickfont=dict(size=axis_tick_font_size),
            title_font=dict(size=axis_title_font_size),
        ),
        width=width,
        height=height,
    )
    if range_color is not None:
        fig.update_coloraxes(
            colorbar_layout(
                range_color[0],
                range_color[1],
                precision=precision,
                title=color_label,
            )
        )
    if x_range is not None:
        fig.update_xaxes(range=x_range)
    if y_range is not None:
        fig.update_yaxes(range=y_range)
    fig.update_traces(marker={"size": marker_size})
    fig.update_coloraxes(colorbar_tickfont_size=colorbar_tickfont_size)

    return fig


def plot_ternary(
    a,
    b,
    c,
    color,
    range_color=None,
    precision=1,
    mask=None,
    a_label="A",
    b_label="B",
    c_label="C",
    color_label="data",
    colorscale="plasma",
    width=1150,
    height=850,
    colorbar_tickfont_size=20,
    ternary_axis_tick_font_size=22,
    ternary_axis_title_font_size=20,
    marker_size=10,
):
    # Convert xarray to 1D numpy array
    if isinstance(a, xr.core.dataarray.DataArray):
        a = a.values.reshape(-1)
    if isinstance(b, xr.core.dataarray.DataArray):
        b = b.values.reshape(-1)
    if isinstance(c, xr.core.dataarray.DataArray):
        c = c.values.reshape(-1)
    if isinstance(color, xr.core.dataarray.DataArray):
        color = color.values.reshape(-1)

    # Convert numpy 2D array to 1D array
    if isinstance(a, np.ndarray):
        if a.ndim > 1:
            a = a.reshape(-1)
    if isinstance(b, np.ndarray):
        if b.ndim > 1:
            b = b.reshape(-1)
    if isinstance(c, np.ndarray):
        if c.ndim > 1:
            c = c.reshape(-1)
    if isinstance(color, np.ndarray):
        if color.ndim > 1:
            color = color.reshape(-1)

    df = pd.DataFrame(
        {
            f"{a_label}": a,
            f"{b_label}": b,
            f"{c_label}": c,
            f"{color_label}": color,
        }
    )
    if mask is not None:
        mask = (df[f"{color_label}"] >= mask[0]) & (df[f"{color_label}"] <= mask[1])
        df = df[mask]

    if range_color is None:
        range_color = (df[f"{color_label}"].min(), df[f"{color_label}"].max())

    fig = px.scatter_ternary(
        df,
        a=f"{a_label}",
        b=f"{b_label}",
        c=f"{c_label}",
        color=f"{color_label}",
        range_color=range_color,
        color_continuous_scale=colorscale,
    )  # Using scatter_ternary from plotly
    fig.update_layout(
        ternary_aaxis=dict(
            tickfont=dict(size=ternary_axis_tick_font_size),
            title_font=dict(size=ternary_axis_title_font_size),
        ),
        ternary_baxis=dict(
            tickfont=dict(size=ternary_axis_tick_font_size),
            title_font=dict(size=ternary_axis_title_font_size),
        ),
        ternary_caxis=dict(
            tickfont=dict(size=ternary_axis_tick_font_size),
            title_font=dict(size=ternary_axis_title_font_size),
        ),
        width=width,
        height=height,
    )
    fig.update_coloraxes(
        colorbar_layout(
            range_color[0],
            range_color[1],
            precision=precision,
            title=color_label,
        )
    )
    fig.update_traces(marker={"size": marker_size})
    fig.update_coloraxes(colorbar_tickfont_size=colorbar_tickfont_size)

    return fig


def plot_waterfall(
    data,
    x,
    y,
    x_label="X",
    y_label="Y",
    x_range=None,
    y_range=None,
    graph_title=None,
    coloraxe_title=None,
    colorscale="plasma",
    range_color=(0, 100),
    width=1050,
    height=700,
    axis_tick_font_size=22,
    axis_title_font_size=20,
    precision=1,
):

    fig = px.imshow(
        data,
        x=x,
        y=y,
        color_continuous_scale=colorscale,
        range_color=range_color,
        aspect="auto",
    )  # using imshow from plotly
    fig.update_layout(
        title=graph_title,
        xaxis_title="X (mm)",
        yaxis_title="Y (mm)",
        width=width,
        height=height,
    )  # Adding title, labels and figure

    fig.update_layout(
        xaxis=dict(
            tickfont=dict(size=axis_tick_font_size),
            title_font=dict(size=axis_title_font_size),
        ),
        yaxis=dict(
            tickfont=dict(size=axis_tick_font_size),
            title_font=dict(size=axis_title_font_size),
        ),
    )
    if x_range is not None:
        fig.update_xaxes(range=x_range)
    if y_range is not None:
        fig.update_yaxes(range=y_range)
    fig.update_yaxes(
        autorange=True
    )  # To have the y axis with negative values below positive values

    return fig


def plot_image(
    data,
    graph_title=None,
    coloraxe_title=None,
    colorscale="plasma",
    range_color=(0, 100),
    width=1000,
    height=1000,
    axis_tick_font_size=22,
    axis_title_font_size=20,
    colorbar_tickfont_size=22,
    colorbar_title_font_size=20,
    precision=0,
    logscale=False,
):
    if logscale:
        plasma = px.colors.sequential.Plasma
        colorscale = [
            [0, plasma[0]],
            [1.0 / 1000, plasma[2]],
            [1.0 / 100, plasma[4]],
            [1.0 / 10, plasma[7]],
            [1.0, plasma[9]],
        ]

    fig = px.imshow(
        data,
        color_continuous_scale=colorscale,
        range_color=range_color,
        aspect="equal",
    )

    fig.update_layout(
        title=graph_title,
        xaxis_title="",
        yaxis_title="",
        width=width,
        height=height,
    )

    fig.update_layout(
        xaxis=dict(
            tickfont=dict(size=axis_tick_font_size),
            title_font=dict(size=axis_title_font_size),
            tickmode="array",
            tickvals=[0, 500, 1000, 1500, 2000],
        ),
        yaxis=dict(
            tickfont=dict(size=axis_tick_font_size),
            title_font=dict(size=axis_title_font_size),
            tickmode="array",
            tickvals=[0, 500, 1000, 1500, 2000],
        ),
    )
    fig.update_coloraxes(
        colorbar_layout(
            range_color[0], range_color[1], precision=precision, title=coloraxe_title
        )
    )
    fig.update_coloraxes(
        colorbar_tickfont_size=colorbar_tickfont_size,
        colorbar_title_font_size=colorbar_title_font_size,
    )

    return fig
