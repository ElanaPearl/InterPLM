import time

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots

PINK_SHADE = "#cc39ca"
CYAN_SHADE = "#00DDFF"


def generate_color_palette(unique_tokens):
    n_colors = len(unique_tokens)
    colors = plt.cm.tab20(np.linspace(0, 1, n_colors))
    aa_to_color = {aa: tuple(color) for aa, color in zip(unique_tokens, colors)}
    return aa_to_color


aa_to_color = generate_color_palette("ACDEFGHIKLMNPQRSTVWY")


def visualize_protein_feature(
    feature_acts,
    sequence,
    metadata,
    characteristic_to_plot="Amino Acids",
    ss_to_full_name=None,
):
    # if feature acts is a torch convert to numpy
    if hasattr(feature_acts, "detach"):
        feature_acts = feature_acts.detach().cpu().numpy()

    fig = make_subplots(rows=1, cols=1)

    # Precompute all colors
    colors = [aa_to_color.get(aa, (200, 200, 200, 1)) for aa in sequence]
    color_strs = [
        f"rgba({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)},{c[3]})" for c in colors
    ]

    for j, (aa, feat) in enumerate(zip(sequence, feature_acts)):
        fig.add_trace(
            go.Scatter(
                x=[j, j],
                y=[0, feat],
                mode="lines",
                line=dict(color="lightgray", width=0.5),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[j],
                y=[feat],
                mode="text",
                text=[aa],
                textfont=dict(size=20, color=color_strs[j]),
                showlegend=False,
                hoverinfo="text",
                hovertext=f"AA: {aa}<br>Position: {j}<br>Activation: {feat:.2f}",
            )
        )

    fig.update_layout(
        xaxis_title="Sequence Position",
        yaxis_title="Feature Activation",
        xaxis=dict(showticklabels=False),
        yaxis=dict(range=[0, max(feature_acts) * 1.1]),
        height=300,
        width=None,
        showlegend=True,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(255,255,255,0.3)",
            font=dict(color="white"),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    return fig


def plot_activations_for_single_feat(feat_acts, feature_to_show):
    if feature_to_show not in feat_acts:
        return None
    acts_for_feat = feat_acts[feature_to_show]
    if len(acts_for_feat) == 0:
        return None
    fig = px.histogram(
        x=acts_for_feat,
        nbins=100,
        labels={"x": "Activation Value", "y": "Count"},
        color_discrete_sequence=[CYAN_SHADE],
    )
    fig.update_layout(
        showlegend=False,
        height=300,
        width=None,
        margin=dict(l=20, r=20, t=0, b=0),
    )
    return fig


def plot_activation_histogram(
    data: np.ndarray,
    feature_to_highlight=None,
    stat="Frequency",
    num_bins: int | str = "auto",
):
    if stat == "Frequency":
        x_title = "Activation Frequency"
    else:
        x_title = f"{stat} Activation Value"

    # Calculate histogram data
    hist, bin_edges = np.histogram(data, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Find example protein indices for each bin
    example_feats = []
    for i in range(len(hist)):
        if i == len(hist) - 1:
            bin_mask = data >= bin_edges[i]
        else:
            bin_mask = (data >= bin_edges[i]) & (data < bin_edges[i + 1])
        example_indices = np.where(bin_mask)[0][:3].tolist()
        example_feats.append(", ".join(map(str, example_indices)))

    # Create hover text
    hover_text = [
        f"{x_title.capitalize()}: {bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}<br>"
        f"Count: {count} features<br>"
        f"Example feat idx: {features}"
        for i, (count, features) in enumerate(zip(hist, example_feats))
    ]

    # Define colors
    histogram_color = PINK_SHADE
    feature_highlight_color = CYAN_SHADE  # "cyan"

    # Create the figure
    fig = go.Figure()

    # Add bar trace
    bar_trace = go.Bar(
        x=bin_centers,
        y=hist,
        hoverinfo="text",
        hovertext=hover_text,
        marker_color=histogram_color,
        width=(bin_edges[1] - bin_edges[0]) * 0.9,
    )
    fig.add_trace(bar_trace)

    # Add value labels
    for i, v in enumerate(hist):
        fig.add_annotation(
            x=bin_centers[i],
            y=v,
            text=str(v),
            showarrow=False,
            yshift=10,
            font=dict(size=10),
        )

    # Add scatter trace for better hover on sparse data
    fig.add_trace(
        go.Scatter(
            x=bin_centers,
            y=hist,
            mode="markers",
            marker=dict(size=10, color="rgba(0,0,0,0)"),
            hoverinfo="text",
            hovertext=hover_text,
            hoverlabel=dict(bgcolor=histogram_color),
        )
    )

    # Determine y-axis ticks dynamically
    y_max = max(hist)
    tick_vals = [1]
    while tick_vals[-1] < y_max:
        tick_vals.append(tick_vals[-1] * 10)

    fig.update_layout(
        title=f"{stat} of Feature Activations",
        xaxis_title=x_title,
        yaxis_title="Count (log scale)",
        yaxis_type="log",
        yaxis=dict(
            tickmode="array",
            tickvals=tick_vals,
            ticktext=[f"{val:,}" for val in tick_vals],
        ),
        hovermode="closest",
        hoverlabel=dict(
            bgcolor=histogram_color,
            font_size=12,
            font_family="Rockwell",
            font_color="black",
        ),
        bargap=0,
        showlegend=False,
        annotations=[
            dict(
                text="",
                xref="paper",
                yref="paper",
                x=0.5,
                y=1.05,
                showarrow=False,
                font=dict(size=10),
            )
        ],
    )

    if feature_to_highlight is not None:
        highlight_value = data[feature_to_highlight]

        # calculate quantile
        sorted_data = np.sort(data)
        idx = np.where(sorted_data == highlight_value)[0][0]
        quantile = idx / len(sorted_data)

        stat_text = (
            f"{highlight_value*100:.2f}%"
            if stat == "Frequency"
            else f"{highlight_value:.2f}"
        )

        annotation_text = (
            f"Feature {feature_to_highlight}<br>"
            + f"   {stat}: {stat_text}<br>"
            + f"   Quantile: {quantile:.2f}"
        )

        fig.add_vline(
            x=highlight_value,
            line=dict(dash="dash", color=feature_highlight_color),
            annotation_text=annotation_text,
            annotation_position=("top right" if quantile < 0.5 else "top left"),
        )

    return fig


def plot_activation_scatter(
    x_value,
    y_value,
    title="",
    xaxis_title="",
    yaxis_title="",
    feature_to_highlight=None,
):
    def get_hover_text(feature_id, x, y):
        return f"<b>f/{feature_id}</b><br>" f" x={x:.0f}%<br>" f"  y={y:.0f}%"

    fig = go.Figure()

    marker_color = PINK_SHADE

    fig.add_trace(
        go.Scatter(
            x=x_value,
            y=y_value,
            mode="markers",
            marker=dict(
                size=5,
                color=marker_color,
                line=dict(width=0, color="DarkSlateGrey"),
            ),
            text=[
                get_hover_text(i, x_value[i], y_value[i]) for i in range(len(x_value))
            ],
            hoverinfo="text",
        )
    )

    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        hovermode="closest",
        showlegend=False,
        height=300,
        width=None,
        margin=dict(l=20, r=20, t=0, b=0),
        autosize=True,
    )

    if feature_to_highlight is not None:
        highlight_x, highlight_y = (
            x_value[feature_to_highlight],
            y_value[feature_to_highlight],
        )
        fig.add_trace(
            go.Scatter(
                x=[highlight_x],
                y=[highlight_y],
                mode="markers",
                marker=dict(size=15, color=CYAN_SHADE),
                showlegend=False,
                hoverinfo="text",
                text=[get_hover_text(feature_to_highlight, highlight_x, highlight_y)],
            ),
        )

    return fig


def plot_structure_scatter(
    df,
    title="",
    xaxis_title="Sequential Effect Size",
    yaxis_title="Structural Effect Size",
    feature_to_highlight=None,
):
    df["sequential_cohen"] = df["sequential_cohen"].apply(lambda x: abs(x))
    df["structural_cohen"] = df["structural_cohen"].apply(lambda x: abs(x))

    def get_hover_text(feature_id, row):
        difference = row["structural_cohen"] / row["sequential_cohen"]
        return (
            f"<b>f/{feature_id}</b><br>"
            f"Structural / Sequential: {difference:.2f}<br>"
        )

    # Calculate difference for coloring
    differences = df["structural_cohen"] / df["sequential_cohen"]
    # differences = differences.apply(lambda x: min(max(0, x), 5))

    # Custom colorscale centered around 0 (where structural = sequential)
    custom_colorscale = [
        [0, "#FFFFFF"],  # Negative difference (structural < sequential)
        [1, PINK_SHADE],  # Positive difference (structural > sequential)
    ]

    fig = go.Figure()

    # Add diagonal line y=x
    min_val = min(df["sequential_cohen"].min(), df["structural_cohen"].min())
    max_val = max(df["sequential_cohen"].max(), df["structural_cohen"].max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(color="gray", dash="dash"),
            name="y=x",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["sequential_cohen"],
            y=df["structural_cohen"],
            mode="markers",
            marker=dict(
                size=5,
                color=differences.apply(lambda x: min(max(0, x), 5)),
                colorscale=custom_colorscale,
                showscale=True,
                line=dict(width=0, color="DarkSlateGrey"),
                colorbar=dict(
                    title="Structural : Sequential Effect Size",
                    titleside="right",
                    tickfont=dict(size=10),
                    titlefont=dict(size=10),
                    tickformat=".2f",  # Decimal format for colorbar
                ),
            ),
            text=[get_hover_text(i, row) for i, row in df.iterrows()],
            hoverinfo="text",
        )
    )

    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        hovermode="closest",
        showlegend=False,
        height=300,
        width=None,
        margin=dict(l=20, r=20, t=0, b=0),
        autosize=True,
    )

    if feature_to_highlight is not None and feature_to_highlight in df.index:
        # breakpoint()
        highlight_row = df.loc[feature_to_highlight]
        fig.add_trace(
            go.Scatter(
                x=[highlight_row["sequential_cohen"]],
                y=[highlight_row["structural_cohen"]],
                mode="markers",
                marker=dict(size=15, color=CYAN_SHADE),
                showlegend=False,
                hoverinfo="text",
                text=[get_hover_text(feature_to_highlight, highlight_row)],
            ),
        )

    return fig


def plot_umap_scatter(
    df,
    title="",
    xaxis_title="UMAP 0",
    yaxis_title="UMAP 1",
    feature_to_highlight=None,
):
    plot_df = df.copy()
    plot_df["cluster"] = plot_df["cluster"].astype(str)

    # Create empty figure
    fig = go.Figure()

    # Create color mapping - make sure we have enough colors
    n_clusters = len(plot_df["cluster"].unique())
    colors = ["lightgray"] + px.colors.sample_colorscale("hsv", n_clusters)

    # Add traces for each cluster
    for cluster in plot_df["cluster"].unique():
        cluster_data = plot_df[plot_df["cluster"] == cluster]

        if cluster == "-1":
            marker_size = 3
            marker_color = "lightgray"
        else:
            marker_size = 4
            cluster_idx = list(sorted(plot_df["cluster"].unique())).index(cluster)
            marker_color = colors[cluster_idx]

        # Create hover text combining Feature and Concept
        hover_text = [
            f"<b>f/{feat}</b><br>Concept: {conc}"
            for feat, conc in zip(cluster_data["Feature"], cluster_data["concept"])
        ]

        fig.add_trace(
            go.Scatter(
                x=cluster_data["UMAP 0"],
                y=cluster_data["UMAP 1"],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=marker_color,
                ),
                opacity=0.6,
                text=hover_text,
                hoverinfo="text",
                showlegend=False,
                name=cluster,
            )
        )

    if feature_to_highlight is not None:  # and feature_to_highlight in df.index:
        # breakpoint()
        highlight_row = df.loc[feature_to_highlight]
        # Add main highlight point
        fig.add_trace(
            go.Scatter(
                x=[highlight_row["UMAP 0"]],
                y=[highlight_row["UMAP 1"]],
                mode="markers",
                marker=dict(
                    size=15,
                    color=CYAN_SHADE,
                    # line=dict(width=2, color='white'),
                ),
                text=[highlight_row["Feature"]],
                hoverinfo="text",
                showlegend=False,
                name="highlight",
            )
        )

    # Update layout
    fig.update_layout(
        showlegend=False,
        height=300,
        width=None,
        margin=dict(l=20, r=20, t=0, b=0),
        autosize=True,
        xaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
        ),
        hovermode="closest",
    )

    return fig
