import networkx as nx
from kloppy import wyscout
from kloppy.domain import Orientation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

df_players = pd.read_csv("data/players.csv", index_col="player_id")

def build_transition_graph(df):
    """
    Build a directed graph of zone transitions from event data

    Args:
        df: DataFrame with 'start_zone_id' and 'end_zone_id' columns
    Returns:
        DiGraph with zones as nodes and transitions as weighted edges
    """
    G = nx.DiGraph()
    transitions = df.dropna(subset=["start_zone_id", "end_zone_id"])
    for _, row in transitions.iterrows():
        start_zone = row["start_zone_id"]
        end_zone = row["end_zone_id"]
        if G.has_edge(start_zone, end_zone):
            G[start_zone][end_zone]["weight"] += 1
        else:
            G.add_edge(start_zone, end_zone, weight=1)

    # add edge attributes
    for u, v in G.edges():
        zone_events = df[(df["start_zone_id"] == u) & (df["end_zone_id"] == v)]
        G[u][v]["transition_frequency"] = G[u][v]["weight"] / G.out_degree(
            u, weight="weight"
        )
        G[u][v]["most_common_event"] = (
            zone_events["event_type"].value_counts().idxmax()
        )  # most common event type
        G[u][v]["start_zone_name"] = zt.get_zone_name(u)
        G[u][v]["end_zone_name"] = zt.get_zone_name(v)

    # add node attributes
    for zone_id in G.nodes():
        zone_events = df[df["start_zone_id"] == zone_id]
        G.nodes[zone_id]["event_count"] = len(zone_events)  # number of events
        G.nodes[zone_id]["most_common_event"] = (
            zone_events["event_type"].value_counts().idxmax()
            if not zone_events.empty
            else None
        )  # most common event type
        G.nodes[zone_id]["unique_players"] = zone_events[
            "player_id"
        ].nunique()  # unique players
        G.nodes[zone_id]["zone_name"] = zt.get_zone_name(zone_id)  # human-readable name
        G.nodes[zone_id]["role_distribution"] = get_role_distributions(
            zone_events
        )  # role distribution [GK, DF, MD, FW]
    return G

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

class ZoneTransformer:
    """Transform pitch coordinates into predefined zones."""

    def __init__(self):
        """Initialize the ZoneTransformer with predefined segments and boundaries."""

        self.INCLUDE_COLS = [
            "event_id",
            "event_type",
            "period_id",
            "timestamp",
            "team_id",
            "player_id",
            "coordinates_*",
            "end_coordinates_*",
            "is_counter_attack",
            "pass_type",
            "result",
            "success",
            "duel_type",
            "set_piece_type",
            "body_part_type",
            "goalkeeper_type",
            "card_type",
        ]

        # self.cols = ["*"]

        self.DROP_COLS = [
            "coordinates_x",
            "coordinates_y",
            "end_coordinates_x",
            "end_coordinates_y",
        ]
        self.width_segments = [6, 10, 17, 17, 17, 17, 10, 6]
        self.height_segments = [19, 18, 26, 18, 19]

        self.n_rows = len(self.height_segments)
        self.n_cols = len(self.width_segments)
        self.n_zones = self.n_rows * self.n_cols  # 8 x 5
        self.outside_zone_id = self.n_zones

        # precompute cumulative boundaries
        self.width_boundaries = np.cumsum([0] + self.width_segments)
        self.height_boundaries = np.cumsum([0] + self.height_segments)

        self.zone_names = self._create_zone_names()

    def _create_zone_names(self) -> dict:
        """Create a mapping from zone IDs to human-readable names."""

        row_names = [
            "RIGHT_WING",
            "RIGHT_HALF",
            "CENTER",
            "LEFT_HALF",
            "LEFT_WING",
        ]

        col_names = [
            "DEF_BOX",  # Defensive 6-yard area
            "DEF_PENALTY",  # Defensive penalty area
            "DEF_THIRD_DEEP",  # Deep defensive third
            "DEF_THIRD",  # Defensive third
            "MID_THIRD_DEF",  # Middle third (defensive half)
            "MID_THIRD_ATT",  # Middle third (attacking half)
            "ATT_THIRD",  # Attacking third
            "ATT_PENALTY",  # Attacking penalty area
        ]

        zone_names = {}
        for col in range(self.n_cols):
            for row in range(self.n_rows):
                zone_id = self.rowcol_to_id(row, col)
                zone_names[zone_id] = f"{row_names[row]}_{col_names[col]}"

        zone_names[self.outside_zone_id] = "OUTSIDE"

        return zone_names

    def coords_to_zone(self, x: float, y: float):
        """Convert (x, y) coordinates to a zone ID and (row, col) tuple.

        Args:
            x (float): x-coordinate (0 to 100)
            y (float): y-coordinate (0 to 100)
        Returns:
            tuple: (zone_id, (row, col))
        """
        if not x or not y or x < 0 or x > 100 or y < 0 or y > 100:
            return self.outside_zone_id, (-1, -1)

        # exactly at boundary (100, 100)
        if x == 100 and y == 100:
            return self.outside_zone_id, (-1, -1)

        # find column
        col = np.searchsorted(self.width_boundaries[1:], x, side="right")
        col = min(col, self.n_cols - 1)

        # find row
        row = np.searchsorted(self.height_boundaries[1:], y, side="right")
        row = min(row, self.n_rows - 1)

        zone_id = self.rowcol_to_id(row, col)
        return zone_id, (row, col)

    def rowcol_to_id(self, row: int, col: int) -> int:
        """Convert (row, col) to zone ID."""
        if row < 0 or row >= self.n_rows or col < 0 or col >= self.n_cols:
            return self.outside_zone_id
        return col * self.n_rows + (self.n_rows - row - 1)

    def id_to_rowcol(self, zone_id: int):
        """Convert zone ID to (row, col)."""
        if zone_id == self.outside_zone_id:
            return (-1, -1)

        row = self.n_rows - (zone_id % self.n_rows) - 1
        col = zone_id // self.n_rows
        return (row, col)

    def get_zone_bounds(self, zone_id: int):
        """Get the (x_min, x_max, y_min, y_max) boundaries of a zone given its ID."""
        if zone_id == self.outside_zone_id:
            return (100.0, 100.0, 100.0, 100.0)

        row, col = self.id_to_rowcol(zone_id)

        x_min = self.width_boundaries[col]
        x_max = self.width_boundaries[col + 1]
        y_min = self.height_boundaries[row]
        y_max = self.height_boundaries[row + 1]

        return (x_min, x_max, y_min, y_max)

    def get_zone_center(self, zone_id: int):
        """Get the (x_center, y_center) of a zone given its ID."""
        if zone_id == self.outside_zone_id:
            return (100.0, 100.0)

        x_min, x_max, y_min, y_max = self.get_zone_bounds(zone_id)
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        return (x_center, y_center)

    def get_zone_name(self, zone_id: int) -> str:
        """Get the human-readable name of a zone given its ID."""
        return self.zone_names.get(zone_id, f"ZONE_{zone_id}")

    def transform_row(self, event):
        start_x = event["coordinates_x"]
        start_y = event["coordinates_y"]

        start_zone_id, start_zone_name = None, None
        if not pd.isna(start_x) and not pd.isna(start_y):
            start_x *= 100
            start_y *= 100
            start_zone_id, _ = self.coords_to_zone(start_x, start_y)
            start_zone_name = self.get_zone_name(start_zone_id)

        end_x = event["end_coordinates_x"]
        end_y = event["end_coordinates_y"]

        end_zone_id, end_zone_name = None, None
        if not pd.isna(end_x) and not pd.isna(end_y):
            end_x *= 100
            end_y *= 100
            end_zone_id, _ = self.coords_to_zone(end_x, end_y)
            end_zone_name = self.get_zone_name(end_zone_id)

        return pd.Series(
            {
                "start_x": start_x,
                "start_y": start_y,
                "start_zone_id": start_zone_id,
                "start_zone_name": start_zone_name,
                "end_x": end_x,
                "end_y": end_y,
                "end_zone_id": end_zone_id,
                "end_zone_name": end_zone_name,
            }
        )

    def transform(self, df):
        """Transform coordinates in the dataframe to zones."""

        zone_data = df.apply(self.transform_row, axis=1)
        df = pd.concat([df, zone_data], axis=1)
        df["start_zone_id"] = df["start_zone_id"].astype(pd.Int64Dtype())
        df["end_zone_id"] = df["end_zone_id"].astype(pd.Int64Dtype())
        df["player_id"] = df["player_id"].astype(pd.Int64Dtype())
        df["team_id"] = df["team_id"].astype(pd.Int64Dtype())

        df = df.drop(columns=self.DROP_COLS, errors="ignore")

        return df
    
def get_role_distributions(zone_events: pd.DataFrame):
    role_counts = df_players[df_players.index.isin(zone_events["player_id"])][
        "role_code2"
    ].value_counts()
    role_distributions = [
        role_counts.get("GK", 0),
        role_counts.get("DF", 0),
        role_counts.get("MD", 0),
        role_counts.get("FW", 0),
    ]
    return (
        np.array(role_distributions) / np.sum(role_distributions)
        if np.sum(role_distributions) > 0
        else np.array([0, 0, 0, 0])
    )

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

zt = ZoneTransformer()

# match_id = 2058017
# data = wyscout.load_open_data(match_id)
# home_team, away_team = data.metadata.teams
# home_team_id, away_team_id = int(home_team.team_id), int(away_team.team_id)

# data = data.transform(
#     Orientation.ACTION_EXECUTING_TEAM
# ) 
# df_raw = data.to_df(*zt.INCLUDE_COLS, engine="pandas")
# df_raw.set_index("event_id", inplace=True)
# df_raw = zt.transform(df_raw)

# INCLUDE_EVENTS = ["PASS", "SHOT", "DUEL", "GOALKEEPER"]
# df = df_raw[df_raw["event_type"].isin(INCLUDE_EVENTS)]
# df_home = df[df["team_id"] == home_team_id]

# G_home = build_transition_graph(df_home)
# visualise_transition_graph(G_home)