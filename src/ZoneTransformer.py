import pandas as pd
import numpy as np


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


zt = ZoneTransformer()
