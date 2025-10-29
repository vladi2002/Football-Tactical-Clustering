# Graph Structure

Each match is represented as a **directed weighted graph** (`nx.DiGraph`) with:

- **Nodes:** 40 fixed pitch zones (+1 for outside)
- **Edges:** Ball transitions between zones

## Node features

| Name                   | Type          | Description                                   |
| ---------------------- | ------------- | --------------------------------------------- |
| `zone_name`            | `str`         | Human-readable zone name                      |
| `x_center`, `y_center` | `float`       | Zone center coordinates (0–100 scale)         |
| `in_deg`, `out_deg`    | `float`       | Weighted in/out degree                        |
| `event_count`          | `int`         | Number of events from this zone               |
| `most_common_event`    | `str`         | Most frequent event type                      |
| `unique_players`       | `int`         | Distinct players acting from this zone        |
| `role_distribution`    | `list[float]` | `[GK, DF, MD, FW]` distribution               |
| `event_distribution`   | `int`         | `[PASS, SHOT, DUEL, GOALKEEPER]` distribution |

## Edge features

| Name                               | Type    | Description                      |
| ---------------------------------- | ------- | -------------------------------- |
| `weight`                           | `int`   | Transition count (`u -> v`)      |
| `transition_frequency`             | `float` | Normalised outgoing transitions  |
| `most_common_event`                | `str`   | Most frequent event type on edge |
| `start_zone_name`, `end_zone_name` | `str`   | Zone labels                      |
