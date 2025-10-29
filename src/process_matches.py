import os
import json
import pickle
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
from kloppy import wyscout
from kloppy.domain import Orientation

from ZoneTransformer import zt

df_players = pd.read_csv("data/players.csv", index_col="player_id")

INCLUDE_EVENTS = ["PASS", "SHOT", "DUEL", "GOALKEEPER"]

GRAPH_DIR = "graphs"
os.makedirs(GRAPH_DIR, exist_ok=True)

COMPETITION_ID = 364  # English Premier League

SELECTED_TEAM_IDS = {
    "Arsenal": 1609,
    "Liverpool": 1612,
    "Manchester City": 1625,
}  # 3 teams


def load_competition(competition_id, matches_path="data/matches.json"):
    """Load all matches from a given competition."""
    with open(matches_path, "r") as f:
        matches = json.load(f)

    comp_matches = [m for m in matches if m["competitionId"] == competition_id]
    return comp_matches


def load_match(match_id):
    """Load match data from Wyscout and transform coordinates."""
    data = wyscout.load_open_data(match_id)
    transformed = data.transform(
        Orientation.ACTION_EXECUTING_TEAM
    )  # the team that executes the action always plays from left to right (with x=100 being the opponent's goal)

    # home_team, away_team = transformed.metadata.teams
    # home_team_id, away_team_id = int(home_team.team_id), int(away_team.team_id)

    df_raw = transformed.to_df(*zt.INCLUDE_COLS, engine="pandas")
    df_raw.set_index("event_id", inplace=True)
    df_raw = zt.transform(df_raw)

    df = df_raw[df_raw["event_type"].isin(INCLUDE_EVENTS)]

    # df_home = df[df["team_id"] == home_team_id]
    # df_away = df[df["team_id"] == away_team_id]

    return df


def build_graph(events):
    """
    Build a directed graph of zone transitions from event data

    Args:
        df: DataFrame with 'start_zone_id' and 'end_zone_id' columns
    Returns:
        DiGraph with zones as nodes and transitions as weighted edges
    """
    G = nx.DiGraph()

    for zone_id in range(zt.n_zones):
        G.add_node(zone_id)

    transitions = events.dropna(subset=["start_zone_id", "end_zone_id"])
    for _, row in transitions.iterrows():
        start_zone = row["start_zone_id"]
        end_zone = row["end_zone_id"]
        event_type = row["event_type"]

        if G.has_edge(start_zone, end_zone):
            G[start_zone][end_zone]["weight"] += 1
            G[start_zone][end_zone]["event_type_counts"][event_type] = (
                G[start_zone][end_zone]["event_type_counts"].get(event_type, 0) + 1
            )
        else:
            G.add_edge(start_zone, end_zone, weight=1)
            G[start_zone][end_zone]["event_type_counts"] = {event_type: 1}

    # add edge attributes
    for u, v in G.edges():
        edge = G[u][v]
        zone_events = events[
            (events["start_zone_id"] == u) & (events["end_zone_id"] == v)
        ]

        out_deg = G.out_degree(u, weight="weight")
        edge["transition_frequency"] = edge["weight"] / out_deg if out_deg > 0 else 0.0

        edge["most_common_event"] = (
            zone_events["event_type"].value_counts().idxmax()
            if not zone_events.empty
            else None
        )  # most common event type
        edge["start_zone_name"] = zt.get_zone_name(u)
        edge["end_zone_name"] = zt.get_zone_name(v)

    # add node attributes
    for zone_id in G.nodes():
        zone = G.nodes[zone_id]
        x_center, y_center = zt.get_zone_center(zone_id)

        zone["zone_name"] = zt.get_zone_name(zone_id)  # human-readable name
        zone["x_center"], zone["y_center"] = x_center, y_center

        zone["in_deg"] = G.in_degree(zone_id, weight="weight")
        zone["out_deg"] = G.out_degree(zone_id, weight="weight")

        out_edges = G.out_edges(zone_id, data=True)
        total_counts = {"PASS": 0, "SHOT": 0, "DUEL": 0, "GOALKEEPER": 0}

        for _, _, edge_data in out_edges:
            for event_type, count in edge_data.get("event_type_counts", {}).items():
                if event_type in total_counts:
                    total_counts[event_type] += count

        total = sum(total_counts.values())
        zone["event_distribution"] = (
            [
                total_counts["PASS"] / total,
                total_counts["SHOT"] / total,
                total_counts["DUEL"] / total,
                total_counts["GOALKEEPER"] / total,
            ]
            if total > 0
            else [0, 0, 0, 0]
        )  # event type distribution [PASS, SHOT, DUEL, GOALKEEPER]

        zone_events = events[events["start_zone_id"] == zone_id]

        zone["unique_players"] = zone_events["player_id"].nunique()  # unique players
        zone["event_count"] = len(zone_events)  # number of events
        zone["most_common_event"] = (
            zone_events["event_type"].value_counts().idxmax()
            if not zone_events.empty
            else None
        )  # most common event type

        role_counts = df_players[df_players.index.isin(zone_events["player_id"])][
            "role_code2"
        ].value_counts()

        role_distribution = [
            role_counts.get("GK", 0),
            role_counts.get("DF", 0),
            role_counts.get("MD", 0),
            role_counts.get("FW", 0),
        ]
        total_roles = sum(role_distribution)
        zone["role_distribution"] = (
            [count / total_roles for count in role_distribution]
            if total_roles > 0
            else [0, 0, 0, 0]
        )  # role distribution [GK, DF, MD, FW]

    return G


def process(
    selected_team_ids,
    competition_id=364,
    matches_path="data/matches.json",
    verbose=False,
):
    """
    Process all matches in a competition.
    For each match:
      - Load event data
      - Build graph
      - Save the graph
    """
    matches = load_competition(competition_id, matches_path)

    for match in tqdm(matches, desc="Processing matches"):
        match_id = match["wyId"]
        teams_data = match["teamsData"]
        team_ids = [t["teamId"] for t in teams_data.values()]

        if not any(tid in selected_team_ids for tid in team_ids):
            continue  # skip matches without selected teams

        try:
            if verbose:
                print(f"\nProcessing match {match_id}...")
            df_match = load_match(match_id)
            if df_match.empty:
                if verbose:
                    print(f"No events for match {match_id}, skipping.")
                continue

            for team_id in team_ids:
                if team_id not in selected_team_ids:
                    continue

                df_team = df_match[df_match["team_id"] == team_id]
                if df_team.empty:
                    continue

                G = build_graph(df_team)

                # save graph
                filename = f"graph_match{match_id}_team{team_id}.pkl"
                filepath = os.path.join(GRAPH_DIR, filename)
                with open(filepath, "wb") as f:
                    pickle.dump(G, f)

                if verbose:
                    print(
                        f"Saved {filename} ({len(G.nodes())} nodes, {len(G.edges())} edges)"
                    )

        except Exception as e:
            if verbose:
                print(f"Skipping match {match_id} due to error: {e}")
            continue


def load_graph(match_id, team_id):
    filename = f"graph_match{match_id}_team{team_id}.pkl"
    filepath = os.path.join(GRAPH_DIR, filename)
    with open(filepath, "rb") as f:
        G = pickle.load(f)
    return G


if __name__ == "__main__":
    # process(
    #     SELECTED_TEAM_IDS.values(),
    #     competition_id=364,
    #     matches_path="data/matches/England.json",
    # )
    G = load_graph(match_id=2499719, team_id=1609)
    print(G.nodes(data=True))
