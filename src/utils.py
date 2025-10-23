import pandas as pd
import numpy as np


def get_role_distributions(df_players: pd.DataFrame, zone_events: pd.DataFrame):
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
