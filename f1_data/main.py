#!/usr/bin/env python3

import argparse
from collections import namedtuple
import time
import dateutil.parser
import functools
from datetime import datetime, timedelta
from typing import Any

import re
import requests
import rich
import numpy
from rich import box
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.prompt import Prompt, FloatPrompt
from rich.table import Table
from rich.text import Text

F1_API = "https://api.openf1.org/v1"

SESSION_2023_BAHRAIN_RACE = "7953"
SESSION_2023_SINGAPORE_RACE = "9158"
SESSION_2023_ABU_DHABI_RACE = "9197"
SESSION_2024_BAHRAIN_QUALI = "9468"


def parse_delta(delta: str) -> timedelta:
    """ Parses a human readable timedelta (3h 05m 19s) into a datetime.timedelta.
    Delta includes:
    * Xh hours
    * Xm minutes
    * Xs seconds
    Values can be negative following timedelta's rules. Eg: -5h-30m

    Based on: https://gist.github.com/santiagobasulto/698f0ff660968200f873a2f9d1c4113c
    Modified to work with h/m/s rather than d/h/m, and accept whitespace
    """

    TIMEDELTA_REGEX = (r'((?P<hours>-?\d+)h)?'
                       r'\s*'
                       r'((?P<minutes>-?\d+)m)?'
                       r'\s*'
                       r'((?P<seconds>-?\d+)s)?')
    TIMEDELTA_PATTERN = re.compile(TIMEDELTA_REGEX, re.IGNORECASE)

    match = TIMEDELTA_PATTERN.match(delta)
    if match:
        parts = {k: int(v) for k, v in match.groupdict().items() if v}
        return timedelta(**parts)


def find_session_key(country_name: str, session_name: str, year: int = 2024) -> str:
    """ Find the session key (ID) for a session via search criteria. """

    rich.print("📁 Downloading Session data...")
    r = requests.get(f"{F1_API}/sessions", params={
        "country_name": country_name,
        "session_name": session_name,
        "year": year,
    })
    r.raise_for_status()
    return r.json()[0]["session_key"]


def get_session_info(session_key: str) -> dict[str, Any]:
    rich.print("📁 Downloading Session data...")
    r = requests.get(f"{F1_API}/sessions", params={
        "session_key": session_key,
    })
    r.raise_for_status()
    data = r.json()[0]
    data["date_start"] = dateutil.parser.parse(data["date_start"])
    data["date_end"] = dateutil.parser.parse(data["date_end"])
    return data


def get_drivers_info(session_key: str) -> dict[int, dict]:
    rich.print("📁 Downloading Driver info...")
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
    rich.print("📁 Downloading Race Position data...")
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
    rich.print("📁 Downloading Pitstop data...")
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
    rich.print("📁 Downloading Stints data...")
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


def get_intervals_by_driver(session_key: str) -> dict[int, dict]:
    rich.print("📁 Downloading Interval data...")
    # date_filter_end = current_time
    # date_filter_begin = current_time - timedelta(seconds=5)
    r = requests.get(
        f"{F1_API}/intervals"
        f"?session_key={session_key}"
        # f"&date>={date_filter_begin}"
        # f"&date<={date_filter_end}"
    )
    r.raise_for_status()
    data = r.json()

    # Sort by timestamp
    sorted_data = sorted(data, key=lambda p: p["date"])

    # Group positions by driver number, preserving date-sorted order
    drivers_intervals = {}
    for item in sorted_data:
        driver = item["driver_number"]
        item["date"] = dateutil.parser.parse(item["date"])
        drivers_intervals.setdefault(driver, []).append(item)

    return drivers_intervals


def get_race_control_data(session_key: str) -> list[dict]:
    """ Get race control information (flags etc). """
    rich.print("📁 Downloading race control data...")
    r = requests.get(f"{F1_API}/race_control", params={
        "session_key": session_key,
    })
    r.raise_for_status()
    data = r.json()
    for event in data:
        event["date"] = dateutil.parser.parse(event["date"])
    return data

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


def render_dashboard(data: dict, race_time: timedelta) -> RenderableType:
    renderables = []

    session_info  = data["session_info"]
    drivers       = data["drivers"]
    all_intervals = data["intervals"]
    race_control  = data["race_control"]
    all_stints    = data["stints"]
    all_pitstops  = data["pitstops"]
    all_positions = data["positions"]

    # Session start time (approximate schedule)
    session_start = session_info["date_start"]

    # Calculate race start time based on green flags
    green_flags = [event for event in race_control if event["flag"] == "GREEN"]
    race_start_time = green_flags[-1]["date"]

    # Calculate race start time based on the earliest position data
    # race_start_time = min([p[0]["date"] for p in all_positions.values()])
    # race_start_delta = timedelta(hours=1, minutes=19, seconds=28)

    current_time = race_start_time + race_time

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
    table.add_column("CHG")                   # Position Change
    table.add_column("Delta")                 # Gap to car ahead
    table.add_column("T")                     # Starting Tyre Compound

    # Filter out pitstops that are in the future
    filtered_pitstops = {driver:[pit for pit in pitstops if pit["date"] <= current_time] for driver, pitstops in all_pitstops.items()}

    # Get the most recent interval for each driver
    current_interval = {
        driver: [interval for interval in intervals if interval["date"] <= current_time][-1:]  # Keep only the most recent interval
        for driver, intervals in all_intervals.items()
    }

    # Dynamic pitstop columns
    most_pitstops = max([len(pitstops) for pitstops in filtered_pitstops.values()] or [0])
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

        change = starting_positions[driver]["position"] - pos
        if change == 0:  pos_str = Text(f"━ {abs(change)}", "grey50")
        elif change > 0: pos_str = Text(f"▲ {abs(change)}", "green")
        elif change < 0: pos_str = Text(f"▼ {abs(change)}", "red")
        row.append(pos_str)

        # Calculate intervals
        interval_list = current_interval[driver]
        interval = (interval_list[0]["interval"] or "-") if interval_list else "-"
        if "LAP" in str(interval): interval_text = Text(f"{interval}", "red1")
        elif "-" in str(interval): interval_text = Text(f"{interval}", "grey70")
        elif interval < 1:         interval_text = Text(f"+{interval}", "green")
        else:                      interval_text = Text(f"+{interval}", "grey70")
        row.append(interval_text)

        pitstops = filtered_pitstops[driver]
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

    # Download data
    session_info = get_session_info(session_key)
    drivers = get_drivers_info(session_key)
    race_control = get_race_control_data(session_key)
    positions = get_positions_by_driver(session_key)
    pitstops = get_pitstops_by_driver(session_key)
    stints = get_stints_by_driver(session_key)
    intervals = get_intervals_by_driver(session_key)

    data = {
        "session_info": session_info,
        "drivers": drivers,
        "race_control": race_control,
        "positions": positions,
        "pitstops": pitstops,
        "stints": stints,
        "intervals": intervals,
    }

    # Manual time sync
    Prompt.pre_prompt("Enter times in format like 1:12:45. Hour can be omitted.")
    video_delta_start =   parse_delta(Prompt.ask("Video time - green flag"))
    video_delta_current = parse_delta(Prompt.ask("Video time - current   "))
    speed_factor = FloatPrompt.ask("Speed Factor (default = 1.0x)", default=1.0)
    race_offset = video_delta_start - video_delta_current

    rich.print(f"{video_delta_start=}")
    rich.print(f"{video_delta_current=}")

    # Event loop
    render_start_time = datetime.now()
    def update_dashboard():
        # Calculate time since start of event loop
        dt = (datetime.now() - render_start_time) * speed_factor

        # Calculate actual time to render the race at
        race_delta = race_offset + dt
        return render_dashboard(data, race_delta)

    with Live(update_dashboard(), refresh_per_second=4) as live:
        while True:
            time.sleep(0.25)
            live.update(update_dashboard())


if __name__ == "__main__":
    main()
