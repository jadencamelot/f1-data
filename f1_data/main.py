#!/usr/bin/env python3

import argparse
import dateutil.parser
import functools
from datetime import datetime, timedelta
from typing import Any

import requests
import rich
import numpy
from rich import box
from rich.console import Console, Group, RenderableType
from rich.table import Table
from rich.text import Text

F1_API = "https://api.openf1.org/v1"

SESSION_2023_BAHRAIN_RACE = "7953"
SESSION_2023_SINGAPORE_RACE = "9158"
SESSION_2023_ABU_DHABI_RACE = "9197"
SESSION_2024_BAHRAIN_QUALI = "9468"

def find_session_key(country_name: str, session_name: str, year: int = 2024) -> str:
    """ Find the session key (ID) for a session via search criteria. """

    print("📁 Downloading Session data...")
    r = requests.get(f"{F1_API}/sessions", params={
        "country_name": country_name,
        "session_name": session_name,
        "year": year,
    })
    r.raise_for_status()
    return r.json()[0]["session_key"]


def get_session_info(session_key: str) -> dict[str, Any]:
    print("📁 Downloading Session data...")
    r = requests.get(f"{F1_API}/sessions", params={
        "session_key": session_key,
    })
    r.raise_for_status()
    data = r.json()[0]
    data["date_start"] = dateutil.parser.parse(data["date_start"])
    data["date_end"] = dateutil.parser.parse(data["date_end"])
    return data


def get_drivers_info(session_key: str) -> dict[int, dict]:
    print("📁 Downloading Driver info...")
    r = requests.get(f"{F1_API}/drivers", params={
        "session_key": session_key,
    })
    r.raise_for_status()
    data = r.json()

    # Group info by driver number
    drivers = {}
    for item in data:
        driver_number = item["driver_number"]
        drivers[driver_number] = item

    return drivers


def get_positions_by_driver(session_key: str) -> dict[int, dict]:
    print("📁 Downloading Race Position data...")
    r = requests.get(f"{F1_API}/position", params={
        "session_key": session_key,
    })
    r.raise_for_status()
    data = r.json()

    # Sort by date
    sorted_data = sorted(data, key=lambda p: p["date"])

    # Group positions by driver number, preserving date sort
    drivers_positions = {}
    for item in sorted_data:
        driver = item["driver_number"]
        drivers_positions.setdefault(driver, []).append({
            "date": dateutil.parser.parse(item["date"]),
            "position": item["position"],
        })

    return drivers_positions


def get_pitstops_by_driver(session_key: str) -> dict[int, dict]:
    print("📁 Downloading Pitstop data...")
    r = requests.get(f"{F1_API}/pit", params={
        "session_key": session_key,
    })
    r.raise_for_status()
    data = r.json()

    # Sort by lap number
    sorted_data = sorted(data, key=lambda p: p["lap_number"])

    # Group positions by driver number, preserving date sort
    drivers_pitstops = {}
    for item in sorted_data:
        driver = item["driver_number"]
        item["date"] = dateutil.parser.parse(item["date"])
        drivers_pitstops.setdefault(driver, []).append(item)

    return drivers_pitstops

def get_stints_by_driver(session_key: str) -> dict[int, dict]:
    print("📁 Downloading Stints data...")
    r = requests.get(f"{F1_API}/stints", params={
        "session_key": session_key,
    })
    r.raise_for_status()
    data = r.json()

    # Sort by lap number
    sorted_data = sorted(data, key=lambda p: p["lap_start"])

    # Group positions by driver number, preserving date sort
    drivers_stints = {}
    for item in sorted_data:
        driver = item["driver_number"]
        drivers_stints.setdefault(driver, []).append(item)

    return drivers_stints


def get_intervals_by_driver(session_key: str, current_time: datetime) -> dict[int, dict]:
    print("📁 Downloading Interval data...")
    date_filter_end = current_time
    date_filter_begin = current_time - timedelta(seconds=5)
    r = requests.get(
        f"{F1_API}/intervals"
        f"?session_key={session_key}"
        f"&date>={date_filter_begin}"
        f"&date<={date_filter_end}"
    )
    r.raise_for_status()
    data = r.json()

    # Sort by timestamp
    sorted_data = sorted(data, key=lambda p: p["date"])

    # Group positions by driver number, keeping only the latest value
    drivers_intervals = {}
    for item in sorted_data:
        driver = item["driver_number"]
        item["date"] = dateutil.parser.parse(item["date"])
        drivers_intervals[driver] = item

    print(f"{len(drivers_intervals)=}")

    return drivers_intervals


def get_race_start_time(session_key: str) -> datetime:
    """ Get the race start time, based on the first lap. """
    print("📁 Downloading first lap data...")
    r = requests.get(f"{F1_API}/stints", params={
        "session_key": session_key,
        "lap_number": 1,
    })
    r.raise_for_status()
    data = r.json()

    # The start date should be the same for all drivers, so just pick any
    data[0]

@functools.cache
def pitstop_quantiles(all_times: tuple[float, ...] ):
    return tuple(numpy.percentile(all_times, [25, 50, 75, 90]))


def pitstop_time_style(time: float, all_times: list[float]) -> str:
    quantiles = pitstop_quantiles(tuple(all_times))
    if   time <= quantiles[0]: return "dark_sea_green"
    elif time <= quantiles[1]: return "dark_sea_green"
    elif time <= quantiles[2]: return "grey53"
    elif time <= quantiles[3]: return "grey53"
    else:                      return "bold red"


def format_pitstop_time(time: float, all_times: list[float]) -> Text:
    quantiles = pitstop_quantiles(tuple(all_times))
    median = quantiles[1]
    time_diff = time - median
    time_str = str(abs(round(time_diff, 2)))
    style = pitstop_time_style(time, all_times)
    if   time_diff > 0: icon = "+"
    elif time_diff < 0: icon = "-"
    else:               icon = " "
    return Text(f"{icon}{time_str}s", style)


def format_tyre_compound(compound: str) -> Text:
    tyre_colour = {
        "H": "grey100",
        "M": "bright_yellow",
        "S": "red",
        "I": "green",
        "W": "blue",
    }
    tyre = compound[0]  # First letter
    return Text(tyre, style=f"bold {tyre_colour[tyre]}")


def render_dashboard(session_key: str, race_time: timedelta) -> RenderableType:
    renderables = []

    # Download data
    drivers = get_drivers_info(session_key)
    session_info = get_session_info(session_key)
    all_positions = get_positions_by_driver(session_key)
    all_pitstops = get_pitstops_by_driver(session_key)
    all_stints = get_stints_by_driver(session_key)

    # Session start time (approximate schedule)
    session_start = session_info["date_start"]

    # Calculate race start time based on the earliest position data
    race_start_time = min([p[0]["date"] for p in all_positions.values()])

    current_time = race_start_time + race_time
    all_intervals = get_intervals_by_driver(session_key, current_time)

    # Starting positions keyed by driver
    starting_positions = {
        driver: positions[0]
        for driver, positions
        in all_positions.items()
    }

    # Calculate grid order at race start and at current point in time
    starting_grid = [-1] * len(drivers)
    current_grid = [-1] * len(drivers)
    for driver, positions in all_positions.items():
        # Filter out position changes that occur later than current time, retaining sort order
        # If the filtered list is empty, just use the first item (starting grid position)
        filtered_positions = [pos for pos in positions if pos["date"] <= current_time] or [positions[0]]
        starting_pos = filtered_positions[0]["position"]
        current_pos = filtered_positions[-1]["position"]

        # Positions start at 1 but list index starts at 0
        starting_grid[starting_pos-1] = driver
        current_grid[current_pos-1] = driver

    # Heading info
    renderables.append(Text(f"         Race time: {race_time}"))
    renderables.append(Text(f"Current time (UTC): {current_time}"))
    renderables.append(Text(f"       Current lap: #TODO"))

    # Format table columns
    table = Table(box=box.SIMPLE, row_styles=["on black", "on grey11"])
    table.add_column("POS", justify="right")  # Position
    table.add_column("DRV", style="bold")     # Driver
    table.add_column("Delta")                 # Gap to car ahead
    table.add_column("CHG")                   # Position Change
    table.add_column("T")                     # Starting Tyre Compound

    # Dynamic pitstop columns
    most_pitstops = max([len(pitstops) for pitstops in all_pitstops.values()] or [0])
    for i in range(most_pitstops):
        table.add_column(f"PIT {i+1}", justify="center")  # Lap

    # Pre-calculate values
    all_pitstop_times = [pitstop["pit_duration"] for pitstops in all_pitstops.values() for pitstop in pitstops]

    for pos, driver in enumerate(current_grid, start=1):
        row = []

        driver_shortname = drivers[driver]["name_acronym"]   # e.g. VER
        driver_colour = f"#{drivers[driver]["team_colour"]}" # e.g. #3671C6
        row.append(Text(f"{pos}."))
        row.append(Text(driver_shortname, driver_colour))

        # TODO - Calculate delta
        delta = "-"
        row.append(delta)

        change = starting_positions[driver]["position"] - pos
        if change == 0:  pos_str = Text(f"━ {abs(change)}", "grey50")
        elif change > 0: pos_str = Text(f"▲ {abs(change)}", "green")
        elif change < 0: pos_str = Text(f"▼ {abs(change)}", "red")
        row.append(pos_str)

        pitstops = all_pitstops[driver]
        stints = all_stints[driver]
        starting_tyre = stints[0]["compound"]
        row.append(format_tyre_compound(starting_tyre))

        # Iterate over every stint AFTER the first one, and the corresponding pitstop
        for pitstop, stint in zip(pitstops, stints[1:]):

            lap = pitstop["lap_number"]
            tyre = stint["compound"]
            time = pitstop["pit_duration"]
            lap_text = Text(str(lap).rjust(2), "grey66")
            tyre_text = format_tyre_compound(tyre)
            time_text = format_pitstop_time(time, all_pitstop_times)
            row.append(Text("  ").join([lap_text, tyre_text, time_text]))

        table.add_row(*row)

    renderables.append(table)
    return Group(*renderables)


def main():
    session_key = SESSION_2023_ABU_DHABI_RACE
    rich.print(f"Session Key: {session_key}")

    dashboard = render_dashboard(session_key, timedelta(minutes=10))

    console = Console()
    console.print(dashboard)


if __name__ == "__main__":
    main()
