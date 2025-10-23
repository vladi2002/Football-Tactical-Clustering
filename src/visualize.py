import networkx as nx
import kloppy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ZoneTransformer import zt


def visualise_transition_graph(G):
    """
    Visualise the zone transition graph.

    Args:
        G: DiGraph with zones as nodes and transitions as weighted edges
    """

    pos = {zone_id: zt.get_zone_center(zone_id) for zone_id in G.nodes()}
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    plt.figure(figsize=(7, 5))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=500,
        node_color="lightblue",
        arrowsize=20,
        edge_color=weights,
        edge_cmap=plt.cm.Blues,
        width=2,
    )
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.Blues, norm=plt.Normalize(vmin=min(weights), vmax=max(weights))
    )
    sm.set_array([])
    # cbar = plt.colorbar(sm, ax=plt.gca(), label="Transition frequency")
    plt.title("Zone transition graph", fontsize=14, fontweight="bold", pad=10)
    plt.show()


def plot_pitch_zones(zt, figsize=(14, 10), highlight_zones=None):
    """
    Plot football pitch with zone grid and IDs

    Args:
        discretizer: FootballFieldDiscretizer instance
        figsize: Figure size (width, height)
        highlight_zones: List of zone IDs to highlight (optional)
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor="#1a1a1a")
    ax.set_facecolor("#2d5016")

    pitch_length = 100
    pitch_width = 100

    # pitch outline
    pitch_rect = patches.Rectangle(
        (0, 0),
        pitch_length,
        pitch_width,
        linewidth=3,
        edgecolor="white",
        facecolor="#2d5016",
        zorder=1,
    )
    ax.add_patch(pitch_rect)

    # halfway line
    ax.plot([50, 50], [0, 100], color="white", linewidth=2, zorder=2)

    # center circle
    center_circle = patches.Circle(
        (50, 50),
        kloppy.domain.WyscoutPitchDimensions.circle_radius,
        linewidth=2,
        edgecolor="white",
        facecolor="none",
        zorder=2,
    )
    ax.add_patch(center_circle)
    ax.plot(50, 50, "o", color="white", markersize=4, zorder=2)

    # penalty areas
    penalty_width = 62
    penalty_depth = 16
    penalty_margin = (100 - penalty_width) / 2

    left_penalty = patches.Rectangle(
        (0, penalty_margin),
        penalty_depth,
        penalty_width,
        linewidth=2,
        edgecolor="white",
        facecolor="none",
        zorder=2,
    )
    ax.add_patch(left_penalty)

    right_penalty = patches.Rectangle(
        (100 - penalty_depth, penalty_margin),
        penalty_depth,
        penalty_width,
        linewidth=2,
        edgecolor="white",
        facecolor="none",
        zorder=2,
    )
    ax.add_patch(right_penalty)

    # goal areas
    goal_width = kloppy.domain.WyscoutPitchDimensions.six_yard_width
    goal_depth = kloppy.domain.WyscoutPitchDimensions.six_yard_length
    goal_margin = (100 - goal_width) / 2

    left_goal = patches.Rectangle(
        (0, goal_margin),
        goal_depth,
        goal_width,
        linewidth=2,
        edgecolor="white",
        facecolor="none",
        zorder=2,
    )
    ax.add_patch(left_goal)

    right_goal = patches.Rectangle(
        (100 - goal_depth, goal_margin),
        goal_depth,
        goal_width,
        linewidth=2,
        edgecolor="white",
        facecolor="none",
        zorder=2,
    )
    ax.add_patch(right_goal)

    # penalty spots
    penalty_spot_distance = kloppy.domain.WyscoutPitchDimensions.penalty_spot_distance
    ax.plot(penalty_spot_distance, 50, "o", color="white", markersize=4, zorder=2)
    ax.plot(100 - penalty_spot_distance, 50, "o", color="white", markersize=4, zorder=2)

    # corner arcs
    corner_radius = kloppy.domain.WyscoutPitchDimensions.corner_radius
    for x, y in [(0, 0), (0, 100), (100, 0), (100, 100)]:
        if x == 0 and y == 0:
            arc = patches.Arc(
                (x, y),
                corner_radius * 2,
                corner_radius * 2,
                angle=0,
                theta1=0,
                theta2=90,
                linewidth=2,
                edgecolor="white",
                zorder=2,
            )
        elif x == 0 and y == 100:
            arc = patches.Arc(
                (x, y),
                corner_radius * 2,
                corner_radius * 2,
                angle=0,
                theta1=270,
                theta2=360,
                linewidth=2,
                edgecolor="white",
                zorder=2,
            )
        elif x == 100 and y == 0:
            arc = patches.Arc(
                (x, y),
                corner_radius * 2,
                corner_radius * 2,
                angle=0,
                theta1=90,
                theta2=180,
                linewidth=2,
                edgecolor="white",
                zorder=2,
            )
        else:  # (100, 100)
            arc = patches.Arc(
                (x, y),
                corner_radius * 2,
                corner_radius * 2,
                angle=0,
                theta1=180,
                theta2=270,
                linewidth=2,
                edgecolor="white",
                zorder=2,
            )
        ax.add_patch(arc)

    for row in range(zt.n_rows):
        for col in range(zt.n_cols):
            zone_id = zt.rowcol_to_id(row, col)
            x_min, x_max, y_min, y_max = zt.get_zone_bounds(zone_id)

            if highlight_zones and zone_id in highlight_zones:
                facecolor = "yellow"
                alpha = 0.3
                edgecolor = "yellow"
                linewidth = 2
            else:
                facecolor = "none"
                alpha = 0.5
                edgecolor = "cyan"
                linewidth = 1

            zone_rect = patches.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=linewidth,
                edgecolor=edgecolor,
                facecolor=facecolor,
                alpha=alpha,
                zorder=3,
            )
            ax.add_patch(zone_rect)

            x_center, y_center = zt.get_zone_center(zone_id)

            bbox_props = dict(
                boxstyle="round,pad=0.3", facecolor="black", alpha=0.7, edgecolor="none"
            )

            ax.text(
                x_center,
                y_center,
                str(zone_id),
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="white",
                bbox=bbox_props,
                zorder=4,
            )

    # ax.text(
    #     50,
    #     -5,
    #     "← DEFENSIVE",
    #     ha="center",
    #     va="top",
    #     fontsize=12,
    #     color="white",
    #     fontweight="bold",
    # )
    ax.text(
        50,
        105,
        "ATTACKING →",
        ha="center",
        va="bottom",
        fontsize=12,
        color="white",
        fontweight="bold",
    )

    info_text = f"Grid: {zt.n_cols} cols x {zt.n_rows} rows = {zt.n_zones} zones"
    ax.text(
        2,
        -3,
        info_text,
        ha="left",
        va="top",
        fontsize=10,
        color="white",
        style="italic",
    )

    ax.set_xlim(-3, 103)
    ax.set_ylim(-8, 108)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.title(
        "Football pitch with zone grids",
        fontsize=16,
        color="white",
        fontweight="bold",
        pad=20,
    )
    plt.tight_layout()

    return fig, ax


def plot_zone_grid_simple(zt, figsize=(14, 10)):
    """
    Simplified grid visualization - just zones and IDs, no pitch markings
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")

    # Draw each zone
    for row in range(zt.n_rows):
        for col in range(zt.n_cols):
            zone_id = zt.rowcol_to_id(row, col)
            x_min, x_max, y_min, y_max = zt.get_zone_bounds(zone_id)

            # alternating colours
            if (row + col) % 2 == 0:
                facecolor = "#e8f4e8"
            else:
                facecolor = "#d4e8d4"

            # Draw zone
            zone_rect = patches.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=2,
                edgecolor="black",
                facecolor=facecolor,
                zorder=1,
            )
            ax.add_patch(zone_rect)

            x_center, y_center = zt.get_zone_center(zone_id)
            zone_name = zt.get_zone_name(zone_id)

            # zone id
            ax.text(
                x_center,
                y_center + 2,
                str(zone_id),
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
                color="black",
            )

            # zone name
            ax.text(
                x_center,
                y_center - 2,
                zone_name.replace("_", "\n"),
                ha="center",
                va="center",
                fontsize=6,
                color="#555",
                style="italic",
            )

    for row in range(zt.n_rows):
        y_center = (zt.height_boundaries[row] + zt.height_boundaries[row + 1]) / 2
        ax.text(
            -2,
            y_center,
            f"Row {row}",
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    for col in range(zt.n_cols):
        x_center = (zt.width_boundaries[col] + zt.width_boundaries[col + 1]) / 2
        ax.text(
            x_center,
            102,
            f"Col {col}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlim(-5, 105)
    ax.set_ylim(-2, 105)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.title("Zone grids", fontsize=14, fontweight="bold", pad=10)
    plt.tight_layout()

    return fig, ax


def plot_zone_heatmap(
    discretizer, zone_counts, figsize=(14, 10), title="Zone activity"
):
    """
    Heatmap visualization of zone activity

    Args:
        discretizer: FootballFieldDiscretizer instance
        zone_counts: dict or Series with {zone_id: count}
        figsize: Figure size
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=figsize)

    if hasattr(zone_counts, "to_dict"):
        zone_counts = zone_counts.to_dict()

    max_count = max(zone_counts.values()) if zone_counts else 1

    for row in range(discretizer.n_rows):
        for col in range(discretizer.n_cols):
            zone_id = discretizer.rowcol_to_id(row, col)
            x_min, x_max, y_min, y_max = discretizer.get_zone_bounds(zone_id)

            count = zone_counts.get(zone_id, 0)
            intensity = count / max_count if max_count > 0 else 0

            # white (no activity) -> red (high activity)
            color = plt.cm.Reds(intensity * 0.8 + 0.2)

            # Draw zone
            zone_rect = patches.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=1,
                edgecolor="gray",
                facecolor=color,
                zorder=1,
            )
            ax.add_patch(zone_rect)

            # labels
            x_center, y_center = discretizer.get_zone_center(zone_id)

            # zone id
            ax.text(
                x_center,
                y_center + 1.5,
                str(zone_id),
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="black",
            )

            # count
            ax.text(
                x_center,
                y_center - 1.5,
                str(count),
                ha="center",
                va="center",
                fontsize=9,
                color="black",
            )

    # add colour bar
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=max_count)
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Event Count", rotation=270, labelpad=15)

    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.title(title, fontsize=14, fontweight="bold", pad=10)
    plt.tight_layout()

    return fig, ax
