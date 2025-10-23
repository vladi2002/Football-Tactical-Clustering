import os
import json
import pickle
import pandas as pd
import numpy as np
import networkx as nx
from kloppy import wyscout
from kloppy.domain import Orientation

from ZoneTransformer import zt
from utils import get_role_distributions

df_players = pd.read_csv("data/players.csv", index_col="player_id")

INCLUDE_EVENTS = ["PASS", "SHOT", "DUEL", "GOALKEEPER"]

GRAPH_DIR = "graphs"
os.makedirs(GRAPH_DIR, exist_ok=True)

COMPETITION_ID = 364  # English Premier League

SELECTED_TEAM_IDS = {
    "Manchester City": 1625,
    "Liverpool": 1612,
    "Arsenal": 1609,
}  # 3 teams to start with


def load_competition(competition_id, matches_path="data/matches.json"):
    """Load all matches from a given competition."""
    with open(matches_path, "r") as f:
        matches = json.load(f)

    comp_matches = [m for m in matches if m["competitionId"] == competition_id]
    print(f"Loaded {len(comp_matches)} matches from competition {competition_id}")
    return comp_matches


def load_match(match_id):
    """Load match data from Wyscout and transform coordinates."""
    data = wyscout.load_open_data(match_id)
    transformed = data.transform(
        Orientation.ACTION_EXECUTING_TEAM
    )  # the team that executes the action always plays from left to right (with x=100 being the opponent's goal)

    home_team, away_team = transformed.metadata.teams
    home_team_id, away_team_id = int(home_team.team_id), int(away_team.team_id)
    print(f"Home team: {home_team.name} ({home_team_id})")
    print(f"Away team: {away_team.name} ({away_team_id})")

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
    transitions = events.dropna(subset=["start_zone_id", "end_zone_id"])
    for _, row in transitions.iterrows():
        start_zone = row["start_zone_id"]
        end_zone = row["end_zone_id"]
        if G.has_edge(start_zone, end_zone):
            G[start_zone][end_zone]["weight"] += 1
        else:
            G.add_edge(start_zone, end_zone, weight=1)

    # add edge attributes
    for u, v in G.edges():
        zone_events = events[
            (events["start_zone_id"] == u) & (events["end_zone_id"] == v)
        ]
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
        zone_events = events[events["start_zone_id"] == zone_id]
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
            df_players, zone_events
        )  # role distribution [GK, DF, MD, FW]
    return G


def process(
    selected_team_ids,
    competition_id=364,
    matches_path="data/matches.json",
):
    """
    Process all matches in a competition.
    For each match:
      - Load event data
      - Build graph
      - Save the graph
    """
    matches = load_competition(competition_id, matches_path)

    for match in matches:
        match_id = match["wyId"]
        teams_data = match["teamsData"]
        team_ids = [t["teamId"] for t in teams_data.values()]

        if not any(tid in selected_team_ids for tid in team_ids):
            continue  # skip matches without selected teams

        try:
            print(f"\nProcessing match {match_id}...")
            df_match = load_match(match_id)
            if df_match.empty:
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

                print(
                    f"Saved {filename} ({len(G.nodes())} nodes, {len(G.edges())} edges)"
                )

        except Exception as e:
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

    print(f"Loaded graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
