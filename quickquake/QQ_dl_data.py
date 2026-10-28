#!/usr/bin/env python3
"""
Download and select waveform data for QuickQuake.

Design:
- process every physical station discovered by QQ_dl_stations.py
- candidate channels were already filtered to seismic data >= 20 Hz
- rank candidate families before waveform download
- try only the best family first
- if unavailable / insufficient coverage / missing response, try next family
- stop after the first valid family
- save every channel at RESAMPLE_RATE (100 Hz, the rate PhaseNet was trained on);
  --resample_rate 0 keeps the native sample rate
- keep exactly one selected representation per physical station
"""

import os
import json
import pickle
import time
import argparse
import copy
from collections import defaultdict

import numpy as np
import obspy
from obspy import UTCDateTime
from obspy.clients.fdsn import Client, RoutingClient
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.io.mseed import InternalMSEEDError


# ============================================================
# SETTINGS
# ============================================================

TARGET_RATE = 100.0
MIN_SAMPLE_RATE = 20.0

# Every saved channel is resampled to this rate (Hz): PhaseNet was trained on
# 100 Hz data, and the QC cannot merge one channel stored at two rates.
# 0 keeps the native rate of each channel.
RESAMPLE_RATE = 100.0

# Require substantial coverage in each selected component.
# Easy to change later if Greenland data contain systematic gaps.
MIN_COVERAGE_FRACTION = 0.80

# Only retry transient server/network failures.
MAX_TRANSIENT_RETRIES = 3
RETRY_SLEEP_SECONDS = 3

VALID_COMPONENTS = {"E", "N", "Z", "1", "2", "3"}

VALID_UNITS = {
    "m/s",
    "m/s**2",
    "m/s^2",
    "m",
}

COMPONENT_ORDER = {
    "E": 0,
    "N": 1,
    "1": 0,
    "2": 1,
    "3": 2,
    "Z": 2,
}

PREFERRED_COMPONENT_SETS = [
    ("E", "N", "Z"),
    ("1", "2", "Z"),
    ("1", "2", "3"),
    ("2", "3", "Z"),
]


# ============================================================
# CLIENTS
# ============================================================

def make_clients(client_name, fallback_names=("EARTHSCOPE",)):
    """
    With the EarthScope federator, also return a list of backup
    data centers, tried in order when the federator has no data
    (e.g. GEOFON serves DK.GDH and ETH serves DK.KARAT).
    """
    if client_name == "EARTHSCOPE_FEDERATOR":
        primary = RoutingClient(
            "earthscope-federator"
        )

        fallback = []

        for name in fallback_names:
            try:
                fallback.append(
                    Client(name)
                )
            except Exception:
                if name == "EARTHSCOPE":
                    fallback.append(
                        Client("IRIS")
                    )
                else:
                    print(
                        "    warning: backup data center "
                        f"{name} is not available"
                    )

        return primary, fallback

    return Client(client_name), []


def is_no_data_error(err):
    if isinstance(
        err,
        FDSNNoDataException,
    ):
        return True

    text = str(err).lower()

    return (
        "204" in text
        or "404" in text
        or "no data available" in text
        or "no data matched request" in text
    )


# ============================================================
# METADATA HELPERS
# ============================================================

def channel_key(
    network,
    station,
    location,
    channel,
):
    return (
        network,
        station,
        location or "",
        channel,
    )


def physical_station_pairs(
    inventory,
):
    return sorted(
        {
            (
                network.code,
                station.code,
            )
            for network in inventory
            for station in network
        }
    )


def station_channels(
    inventory,
    network_code,
    station_code,
):
    output = []

    for network in inventory:
        if network.code != network_code:
            continue

        for station in network:
            if station.code != station_code:
                continue

            output.extend(
                list(station.channels)
            )

    return output


def normalize_unit(unit):
    return (
        str(unit or "")
        .strip()
        .lower()
        .replace(" ", "")
    )


def response_info(channel):
    if channel is None:
        return None, None

    try:
        sensitivity = (
            channel.response
            .instrument_sensitivity
        )

        value = float(
            sensitivity.value
        )

        unit = normalize_unit(
            sensitivity.input_units
        )

    except Exception:
        return None, None

    if value == 0:
        return None, None

    if unit not in VALID_UNITS:
        return None, None

    return value, unit


# ============================================================
# CANDIDATE FAMILY RANKING
# ============================================================

def choose_component_set(
    components,
):
    """
    Prefer 3C, then 2C, then 1C.
    """
    available = set(
        components
    )

    for triplet in (
        PREFERRED_COMPONENT_SETS
    ):
        if set(triplet).issubset(
            available
        ):
            return tuple(
                triplet
            )

    # Prefer 2C sets containing Z.
    two_component_preferences = [
        ("E", "Z"),
        ("N", "Z"),
        ("1", "Z"),
        ("2", "Z"),
        ("1", "2"),
        ("E", "N"),
        ("2", "3"),
        ("1", "3"),
    ]

    for pair in two_component_preferences:
        if set(pair).issubset(
            available
        ):
            return tuple(
                pair
            )

    # Last resort: one component.
    for component in (
        "Z",
        "3",
        "E",
        "N",
        "1",
        "2",
    ):
        if component in available:
            return (
                component,
            )

    return tuple()


def rate_score(rate):
    """
    Lower is better.

    Preference:
      1) exactly 100 Hz
      2) >100 Hz, lowest downsampling factor first
      3) 20-99 Hz, highest rate first

    1 Hz / 0.1 Hz channels never arrive here because
    QQ_dl_stations.py already removed them.
    """
    rate = float(rate)

    if abs(
        rate - TARGET_RATE
    ) < 0.1:
        return (
            0,
            0.0,
        )

    if rate > TARGET_RATE:
        return (
            1,
            rate - TARGET_RATE,
        )

    return (
        2,
        TARGET_RATE - rate,
    )


def component_score(
    components,
):
    n = len(
        components
    )

    if n >= 3:
        return 0

    if n == 2:
        return 1

    if n == 1:
        return 2

    return 9


def build_family_candidates(
    inventory,
    network_code,
    station_code,
):
    """
    Build metadata candidates grouped by:
        physical station + location + family

    No waveform is downloaded here.
    """
    groups = defaultdict(
        list
    )

    for channel in station_channels(
        inventory,
        network_code,
        station_code,
    ):
        code = channel.code or ""

        if (
            len(code) < 3
            or code[-1]
            not in VALID_COMPONENTS
        ):
            continue

        try:
            rate = float(
                channel.sample_rate
            )
        except Exception:
            continue

        if rate < MIN_SAMPLE_RATE:
            continue

        location = (
            channel.location_code
            or ""
        )

        family = code[:-1]

        groups[
            (
                location,
                family,
            )
        ].append(
            channel
        )

    candidates = []

    for (
        location,
        family,
    ), channels in groups.items():

        # One metadata entry per component.
        by_component = defaultdict(
            list
        )

        for channel in channels:
            by_component[
                channel.code[-1]
            ].append(
                channel
            )

        components = (
            choose_component_set(
                by_component.keys()
            )
        )

        if not components:
            continue

        chosen_metadata = []

        # Prefer the highest metadata sample rate if duplicated.
        for component in components:
            options = sorted(
                by_component[
                    component
                ],
                key=lambda channel:
                    float(
                        channel.sample_rate
                    ),
                reverse=True,
            )

            chosen_metadata.append(
                options[0]
            )

        rates = [
            float(
                channel.sample_rate
            )
            for channel
            in chosen_metadata
        ]

        # Conservative score uses the lowest selected rate.
        candidate_rate = min(
            rates
        )

        candidates.append(
            {
                "network":
                    network_code,
                "station":
                    station_code,
                "location":
                    location,
                "family":
                    family,
                "components":
                    components,
                "channels": [
                    family
                    + component
                    for component
                    in components
                ],
                "metadata":
                    chosen_metadata,
                "metadata_rates":
                    rates,
                "candidate_rate":
                    candidate_rate,
                "score": (
                    component_score(
                        components
                    ),
                    rate_score(
                        candidate_rate
                    ),
                    0
                    if location == ""
                    else 1,
                    family,
                    location,
                ),
            }
        )

    return sorted(
        candidates,
        key=lambda candidate:
            candidate["score"],
    )


# ============================================================
# WAVEFORM REQUESTS
# ============================================================

def fdsn_location(location):
    # FDSN selector for blank location code.
    return (
        location
        if location
        else "--"
    )


def request_waveforms_once(
    client,
    candidate,
    start,
    end,
):
    return client.get_waveforms(
        network=candidate[
            "network"
        ],
        station=candidate[
            "station"
        ],
        location=fdsn_location(
            candidate[
                "location"
            ]
        ),
        channel=",".join(
            candidate[
                "channels"
            ]
        ),
        starttime=start,
        endtime=end,
    )


def request_waveforms(
    client,
    candidate,
    start,
    end,
):
    """
    204/404 => immediate no-data result.
    Only transient errors are retried.
    """
    for attempt in range(
        1,
        MAX_TRANSIENT_RETRIES + 1,
    ):
        try:
            return request_waveforms_once(
                client=client,
                candidate=candidate,
                start=start,
                end=end,
            )

        except Exception as err:
            if is_no_data_error(err):
                return obspy.Stream()

            print(
                f"    transient waveform error "
                f"({attempt}/"
                f"{MAX_TRANSIENT_RETRIES}): "
                f"{err}"
            )

            if (
                attempt
                < MAX_TRANSIENT_RETRIES
            ):
                time.sleep(
                    RETRY_SLEEP_SECONDS
                )

    return obspy.Stream()


def download_candidate(
    primary,
    fallback,
    candidate,
    start,
    end,
):
    stream = request_waveforms(
        client=primary,
        candidate=candidate,
        start=start,
        end=end,
    )

    # Backup data centers, in order, until one returns data
    for client in fallback:
        if len(stream) > 0:
            break

        stream = request_waveforms(
            client=client,
            candidate=candidate,
            start=start,
            end=end,
        )

        if len(stream) > 0:
            print(
                "    backup data center "
                f"{client.base_url} returned data"
            )

    return stream


# ============================================================
# COVERAGE
# ============================================================

def trace_coverage_seconds(
    traces,
    start,
    end,
):
    intervals = []

    for trace in traces:
        if (
            trace.stats.npts
            <= 0
        ):
            continue

        trace_start = max(
            trace.stats.starttime,
            start,
        )

        trace_end = min(
            trace.stats.endtime,
            end,
        )

        if trace_end <= trace_start:
            continue

        intervals.append(
            (
                trace_start,
                trace_end,
            )
        )

    if not intervals:
        return 0.0

    intervals.sort(
        key=lambda item:
            item[0]
    )

    merged = []

    current_start, current_end = (
        intervals[0]
    )

    for (
        next_start,
        next_end,
    ) in intervals[1:]:

        if next_start <= current_end:
            current_end = max(
                current_end,
                next_end,
            )
        else:
            merged.append(
                (
                    current_start,
                    current_end,
                )
            )

            current_start = (
                next_start
            )

            current_end = (
                next_end
            )

    merged.append(
        (
            current_start,
            current_end,
        )
    )

    return sum(
        float(
            b - a
        )
        for a, b
        in merged
    )


def evaluate_waveform(
    candidate,
    stream,
    start,
    end,
):
    """
    Confirm actual components, actual rates and coverage.

    The candidate is accepted only if every selected component
    has >= MIN_COVERAGE_FRACTION of the requested time window.
    """
    window_seconds = max(
        float(
            end - start
        ),
        1.0,
    )

    by_component = defaultdict(
        list
    )

    for trace in stream:
        code = (
            trace.stats.channel
            or ""
        )

        if (
            not code
            or code[-1]
            not in VALID_COMPONENTS
        ):
            continue

        # Keep only the requested family.
        if (
            code[:-1]
            != candidate[
                "family"
            ]
        ):
            continue

        # Keep only requested location.
        if (
            (trace.stats.location or "")
            != candidate[
                "location"
            ]
        ):
            continue

        if (
            float(
                trace.stats.sampling_rate
            )
            < MIN_SAMPLE_RATE
        ):
            continue

        by_component[
            code[-1]
        ].append(
            trace
        )

    actual_components = (
        choose_component_set(
            by_component.keys()
        )
    )

    if not actual_components:
        return None

    coverage_by_component = {}
    rate_by_component = {}

    for component in actual_components:
        traces = by_component[
            component
        ]

        coverage_seconds = (
            trace_coverage_seconds(
                traces=traces,
                start=start,
                end=end,
            )
        )

        coverage_fraction = min(
            coverage_seconds
            / window_seconds,
            1.0,
        )

        coverage_by_component[
            component
        ] = coverage_fraction

        rates = {
            float(
                trace.stats.sampling_rate
            )
            for trace in traces
        }

        # A normal family should have one actual rate.
        # Keep the highest one if segmented metadata differ.
        rate_by_component[
            component
        ] = max(
            rates
        )

    minimum_coverage = min(
        coverage_by_component.values()
    )

    if (
        minimum_coverage
        < MIN_COVERAGE_FRACTION
    ):
        return {
            "accepted":
                False,
            "reason":
                "insufficient_coverage",
            "components":
                actual_components,
            "coverage_by_component":
                coverage_by_component,
            "rate_by_component":
                rate_by_component,
        }

    selected_traces = []

    for component in actual_components:
        selected_traces.extend(
            by_component[
                component
            ]
        )

    return {
        "accepted":
            True,
        "reason":
            "waveform_ok",
        "components":
            actual_components,
        "coverage_by_component":
            coverage_by_component,
        "minimum_coverage":
            minimum_coverage,
        "rate_by_component":
            rate_by_component,
        "traces":
            selected_traces,
    }


# ============================================================
# RESPONSE METADATA
# ============================================================

def request_response_inventory_once(
    client,
    candidate,
    components,
    start,
    end,
):
    channels = [
        candidate["family"]
        + component
        for component
        in components
    ]

    return client.get_stations(
        network=candidate[
            "network"
        ],
        station=candidate[
            "station"
        ],
        location=fdsn_location(
            candidate[
                "location"
            ]
        ),
        channel=",".join(
            channels
        ),
        starttime=start,
        endtime=end,
        level="response",
    )


def request_response_inventory(
    primary,
    fallback,
    candidate,
    components,
    start,
    end,
):
    inventory = None

    try:
        inventory = (
            request_response_inventory_once(
                client=primary,
                candidate=candidate,
                components=components,
                start=start,
                end=end,
            )
        )
    except Exception as err:
        if not is_no_data_error(err):
            print(
                "    response metadata error: "
                f"{err}"
            )

    # Backup data centers, in order, until one returns the response
    for client in fallback:
        if inventory is not None:
            break

        try:
            inventory = (
                request_response_inventory_once(
                    client=client,
                    candidate=candidate,
                    components=components,
                    start=start,
                    end=end,
                )
            )
        except Exception as err:
            if not is_no_data_error(err):
                print(
                    "    response metadata error "
                    f"({client.base_url}): {err}"
                )

    return inventory


def response_lookup(
    inventory,
):
    output = {}

    if inventory is None:
        return output

    for network in inventory:
        for station in network:
            for channel in station:
                output[
                    channel_key(
                        network.code,
                        station.code,
                        channel.location_code,
                        channel.code,
                    )
                ] = channel

    return output


def validate_response(
    candidate,
    waveform_result,
    response_inventory,
):
    lookup = response_lookup(
        response_inventory
    )

    response_values = []
    response_units = []
    response_channels = []

    for component in (
        waveform_result[
            "components"
        ]
    ):
        code = (
            candidate["family"]
            + component
        )

        metadata = lookup.get(
            channel_key(
                candidate[
                    "network"
                ],
                candidate[
                    "station"
                ],
                candidate[
                    "location"
                ],
                code,
            )
        )

        value, unit = (
            response_info(
                metadata
            )
        )

        if (
            value is None
            or unit is None
        ):
            return None

        response_values.append(
            value
        )

        response_units.append(
            unit
        )

        response_channels.append(
            metadata
        )

    if len(
        set(response_units)
    ) != 1:
        return None

    return {
        "responses":
            response_values,
        "unit":
            response_units[0],
        "metadata":
            response_channels,
    }


# ============================================================
# SELECT ONE FAMILY PER PHYSICAL STATION
# ============================================================

def select_station(
    primary,
    fallback,
    inventory,
    network_code,
    station_code,
    start,
    end,
):
    candidates = (
        build_family_candidates(
            inventory=inventory,
            network_code=network_code,
            station_code=station_code,
        )
    )

    attempts = []

    if not candidates:
        return None, attempts

    print(
        f"\n[waveforms] "
        f"{network_code}.{station_code}"
    )

    for rank, candidate in enumerate(
        candidates,
        start=1,
    ):
        location_print = (
            candidate["location"]
            if candidate["location"]
            else "--"
        )

        print(
            f"  try #{rank}: "
            f"loc={location_print} "
            f"family={candidate['family']} "
            f"components="
            f"{','.join(candidate['components'])} "
            f"metadata_rates="
            f"{candidate['metadata_rates']} Hz"
        )

        stream = download_candidate(
            primary=primary,
            fallback=fallback,
            candidate=candidate,
            start=start,
            end=end,
        )

        if len(stream) == 0:
            print(
                "    -> no waveform"
            )

            attempts.append(
                {
                    "family":
                        candidate["family"],
                    "location":
                        candidate["location"],
                    "result":
                        "no_waveform",
                }
            )

            continue

        waveform_result = (
            evaluate_waveform(
                candidate=candidate,
                stream=stream,
                start=start,
                end=end,
            )
        )

        if waveform_result is None:
            print(
                "    -> no usable components"
            )

            attempts.append(
                {
                    "family":
                        candidate["family"],
                    "location":
                        candidate["location"],
                    "result":
                        "no_usable_components",
                }
            )

            continue

        if not waveform_result[
            "accepted"
        ]:
            coverage_text = ", ".join(
                f"{component}="
                f"{100.0 * fraction:.1f}%"
                for component, fraction
                in waveform_result[
                    "coverage_by_component"
                ].items()
            )

            print(
                "    -> insufficient coverage: "
                f"{coverage_text}"
            )

            attempts.append(
                {
                    "family":
                        candidate["family"],
                    "location":
                        candidate["location"],
                    "result":
                        "insufficient_coverage",
                    "coverage":
                        waveform_result[
                            "coverage_by_component"
                        ],
                }
            )

            continue

        response_inventory = (
            request_response_inventory(
                primary=primary,
                fallback=fallback,
                candidate=candidate,
                components=(
                    waveform_result[
                        "components"
                    ]
                ),
                start=start,
                end=end,
            )
        )

        response_result = (
            validate_response(
                candidate=candidate,
                waveform_result=waveform_result,
                response_inventory=(
                    response_inventory
                ),
            )
        )

        if response_result is None:
            print(
                "    -> waveform exists, "
                "but valid response metadata "
                "was not available"
            )

            attempts.append(
                {
                    "family":
                        candidate["family"],
                    "location":
                        candidate["location"],
                    "result":
                        "missing_response",
                }
            )

            continue

        components = (
            waveform_result[
                "components"
            ]
        )

        channels = [
            candidate["family"]
            + component
            for component
            in components
        ]

        actual_rates = [
            waveform_result[
                "rate_by_component"
            ][component]
            for component
            in components
        ]

        chosen = {
            "network":
                network_code,
            "station":
                station_code,
            "location":
                candidate["location"],
            "family":
                candidate["family"],
            "components":
                components,
            "channels":
                channels,
            "rates":
                actual_rates,
            "coverage_fraction":
                waveform_result[
                    "minimum_coverage"
                ],
            "traces":
                waveform_result[
                    "traces"
                ],
            "responses":
                response_result[
                    "responses"
                ],
            "unit":
                response_result[
                    "unit"
                ],
            "metadata":
                response_result[
                    "metadata"
                ],
            "response_inventory":
                response_inventory,
        }

        print(
            "    -> SELECTED "
            f"{','.join(channels)} "
            f"rates={actual_rates} Hz "
            f"coverage="
            f"{100.0 * chosen['coverage_fraction']:.1f}%"
        )

        attempts.append(
            {
                "family":
                    candidate["family"],
                "location":
                    candidate["location"],
                "result":
                    "selected",
                "channels":
                    channels,
                "rates":
                    actual_rates,
                "coverage_fraction":
                    chosen[
                        "coverage_fraction"
                    ],
            }
        )

        return chosen, attempts

    print(
        "    -> station rejected: "
        "no candidate family passed all checks"
    )

    return None, attempts


# ============================================================
# FINAL OUTPUT METADATA
# ============================================================

def build_station_json(
    selected_candidates,
):
    output = {}

    for selected in (
        selected_candidates
    ):
        ordered = sorted(
            zip(
                selected[
                    "components"
                ],
                selected[
                    "responses"
                ],
                selected[
                    "metadata"
                ],
            ),
            key=lambda item:
                COMPONENT_ORDER.get(
                    item[0],
                    9,
                ),
        )

        first_channel = (
            ordered[0][2]
        )

        sid = (
            f"{selected['network']}."
            f"{selected['station']}."
            f"{selected['location']}."
            f"{selected['family']}"
        )

        output[sid] = {
            "longitude":
                float(
                    first_channel.longitude
                ),
            "latitude":
                float(
                    first_channel.latitude
                ),
            "elevation(m)":
                float(
                    first_channel.elevation
                ),
            "component": [
                item[0]
                for item
                in ordered
            ],
            "response": [
                float(
                    item[1]
                )
                for item
                in ordered
            ],
            "unit":
                selected["unit"],
        }

    return output


def combine_selected_inventories(
    selected_candidates,
):
    combined = None

    for selected in selected_candidates:
        inventory = (
            selected[
                "response_inventory"
            ]
        )

        if inventory is None:
            continue

        if combined is None:
            combined = copy.deepcopy(
                inventory
            )
        else:
            combined += copy.deepcopy(
                inventory
            )

    return combined


# ============================================================
# RESAMPLING
# ============================================================

def resample_stream(
    stream,
    rate,
):
    """
    Bring every trace of the stream to `rate` Hz, in place.

    - faster than `rate`: low-pass at 0.4 * rate first (anti-alias), then
      interpolate; without the filter, energy above the new Nyquist
      frequency would fold back as false low frequencies
    - slower than `rate`: interpolate only; this adds samples but no
      signal above the native Nyquist frequency
    - within 0.01 % of `rate` (clock jitter, e.g. 99.99994659 Hz): interpolate
    - rate <= 0: nothing is changed

    Interpolation is Lanczos (a = 20), which keeps the 1-15 Hz amplitudes.
    ObsPy's "weighted_average_slopes" (used by seisgo) keeps only 77 % of a
    15 Hz wave recorded at 40 Hz, and on short segments it can return a
    garbage last sample (e.g. 4.7e200).

    Integer counts stay integers, so MiniSEED keeps its compression.

    Returns {trace id: native rate} for the upsampled traces.
    """
    upsampled = {}

    if not rate or rate <= 0:
        return upsampled

    for trace in list(
        stream
    ):
        native = float(
            trace.stats.sampling_rate
        )

        if native == rate:
            continue

        dtype = trace.data.dtype

        try:
            if native > rate * (1 + 1e-4):
                trace.filter(
                    "lowpass",
                    freq=0.4 * rate,
                    corners=8,
                    zerophase=True,
                )
            elif native < rate * (1 - 1e-4):
                upsampled[
                    trace.id
                ] = native

            trace.interpolate(
                sampling_rate=rate,
                method="lanczos",
                a=20,
            )
        except Exception as err:
            # e.g. a segment of one sample cannot be interpolated
            print(
                f"    warning: dropping a segment of {trace.id} "
                f"({trace.stats.npts} samples) that could not be "
                f"resampled: {err}"
            )
            stream.remove(
                trace
            )
            continue

        if np.issubdtype(
            dtype,
            np.integer,
        ):
            trace.data = np.round(
                trace.data
            ).astype(
                dtype
            )

    return upsampled


# ============================================================
# MAIN
# ============================================================

def download_waveforms(
    config_json,
    output_dir,
    fallback_names=("EARTHSCOPE",),
):
    with open(config_json) as fp:
        config = json.load(fp)

    client_name = config.get(
        "client",
        "EARTHSCOPE_FEDERATOR",
    )

    primary, fallback = (
        make_clients(
            client_name,
            fallback_names,
        )
    )

    print(
        f"[waveforms] FDSN client = "
        f"{client_name}"
        + (f" | backups: {', '.join(c.base_url for c in fallback)}" if fallback else "")
    )

    stations_path = os.path.join(
        output_dir,
        "stations_candidates.pkl",
    )

    if not os.path.exists(
        stations_path
    ):
        # Compatibility fallback.
        stations_path = os.path.join(
            output_dir,
            "stations.pkl",
        )

    if not os.path.exists(
        stations_path
    ):
        raise RuntimeError(
            "Missing station candidate metadata."
        )

    with open(
        stations_path,
        "rb",
    ) as fp:
        stations = pickle.load(fp)

    pairs = physical_station_pairs(
        stations
    )

    start = UTCDateTime(
        config["starttime"]
    )

    end = UTCDateTime(
        config["endtime"]
    )

    waveform_dir = os.path.join(
        output_dir,
        "waveforms",
    )

    os.makedirs(
        waveform_dir,
        exist_ok=True,
    )

    fname = (
        start.datetime.strftime(
            "%Y-%m-%dT%H-%M-%S"
        )
        + ".mseed"
    )

    outpath = os.path.join(
        waveform_dir,
        fname,
    )

    input_csv = os.path.join(
        output_dir,
        "input_data.csv",
    )

    # Remove stale downstream inputs before selection.
    if os.path.exists(
        input_csv
    ):
        os.remove(
            input_csv
        )

    if os.path.exists(
        outpath
    ):
        os.remove(
            outpath
        )

    final_stream = obspy.Stream()
    selected_candidates = []
    diagnostics = []

    print(
        "\n[waveforms] Candidate physical stations: "
        f"{len(pairs)}"
    )

    # --------------------------------------------------------
    # One physical station at a time.
    # --------------------------------------------------------

    for network_code, station_code in pairs:
        selected, attempts = (
            select_station(
                primary=primary,
                fallback=fallback,
                inventory=stations,
                network_code=network_code,
                station_code=station_code,
                start=start,
                end=end,
            )
        )

        diagnostics.append(
            {
                "station":
                    f"{network_code}."
                    f"{station_code}",
                "selected": (
                    None
                    if selected is None
                    else {
                        "location":
                            selected[
                                "location"
                            ],
                        "family":
                            selected[
                                "family"
                            ],
                        "channels":
                            selected[
                                "channels"
                            ],
                        "rates":
                            selected[
                                "rates"
                            ],
                        "coverage_fraction":
                            selected[
                                "coverage_fraction"
                            ],
                    }
                ),
                "attempts":
                    attempts,
            }
        )

        if selected is None:
            continue

        chosen_stream = obspy.Stream(
            traces=[
                trace.copy()
                for trace
                in selected[
                    "traces"
                ]
            ]
        )

        # Resample before merging, so a channel that changed rate
        # within the chunk merges into one trace.
        upsampled = resample_stream(
            chosen_stream,
            RESAMPLE_RATE,
        )

        for trace_id, native in sorted(
            upsampled.items()
        ):
            print(
                f"    warning: {trace_id} is sampled at "
                f"{native:g} Hz, below {RESAMPLE_RATE:g} Hz; "
                f"upsampled by interpolation (no signal above "
                f"{native / 2:g} Hz)"
            )

        try:
            chosen_stream.merge(
                method=1,
                fill_value=0,
            )
        except Exception as err:
            print(
                f"    warning: merge failed "
                f"for {network_code}."
                f"{station_code}: {err}"
            )

        final_stream += chosen_stream

        selected_candidates.append(
            selected
        )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    with open(
        os.path.join(
            output_dir,
            "channel_selection.json",
        ),
        "w",
    ) as fp:
        json.dump(
            diagnostics,
            fp,
            indent=2,
        )

    if not selected_candidates:
        raise RuntimeError(
            "No physical station passed waveform, "
            "coverage and response checks."
        )

    # --------------------------------------------------------
    # Save selected waveform data.
    # --------------------------------------------------------

    try:
        final_stream.write(
            outpath,
            format="MSEED",
        )
    except InternalMSEEDError:
        final_stream.write(
            outpath,
            format="MSEED",
            encoding="INT32",
        )

    # --------------------------------------------------------
    # Save final PhaseNet/GaMMA station metadata.
    # --------------------------------------------------------

    station_json = (
        build_station_json(
            selected_candidates
        )
    )

    with open(
        os.path.join(
            output_dir,
            "stations.json",
        ),
        "w",
    ) as fp:
        json.dump(
            station_json,
            fp,
            indent=2,
        )

    selected_inventory = (
        combine_selected_inventories(
            selected_candidates
        )
    )

    if selected_inventory is None:
        raise RuntimeError(
            "Selected stations did not produce "
            "a final response inventory."
        )

    selected_inventory.write(
        os.path.join(
            output_dir,
            "stations.xml",
        ),
        format="STATIONXML",
    )

    with open(
        os.path.join(
            output_dir,
            "stations.pkl",
        ),
        "wb",
    ) as fp:
        pickle.dump(
            selected_inventory,
            fp,
        )

    with open(
        input_csv,
        "w",
    ) as fp:
        fp.write(
            "fname\n"
        )
        fp.write(
            f"{fname}\n"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    rate_counts = defaultdict(
        int
    )

    component_counts = defaultdict(
        int
    )

    for selected in (
        selected_candidates
    ):
        # Report representative minimum actual rate.
        representative_rate = min(
            selected["rates"]
        )

        rate_counts[
            representative_rate
        ] += 1

        component_counts[
            len(
                selected[
                    "components"
                ]
            )
        ] += 1

    print(
        "\n"
        + "=" * 70
    )

    print(
        "[waveforms] FINAL SELECTION SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        "Candidate physical stations: "
        f"{len(pairs)}"
    )

    print(
        "Selected physical stations:  "
        f"{len(selected_candidates)}"
    )

    print(
        "Rejected physical stations:  "
        f"{len(pairs) - len(selected_candidates)}"
    )

    print(
        "\nSelected by component count:"
    )

    for n_components in sorted(
        component_counts,
        reverse=True,
    ):
        print(
            f"  {n_components}C: "
            f"{component_counts[n_components]}"
        )

    print(
        "\nSelected by sample rate:"
    )

    for rate in sorted(
        rate_counts,
        reverse=True,
    ):
        print(
            f"  {rate:g} Hz: "
            f"{rate_counts[rate]}"
        )

    print(
        "Saved at: "
        + (
            f"{RESAMPLE_RATE:g} Hz"
            if RESAMPLE_RATE > 0
            else "native sample rates"
        )
    )

    print(
        f"\nWaveforms saved at: "
        f"{outpath}"
    )

    print(
        "[waveforms] stations.json contains "
        "only selected physical stations."
    )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        required=True,
    )

    parser.add_argument(
        "--target_rate",
        type=float,
        default=TARGET_RATE,
        help=f"Preferred sample rate when ranking channel families, in Hz (default: {TARGET_RATE})",
    )

    parser.add_argument(
        "--min_sample_rate",
        type=float,
        default=MIN_SAMPLE_RATE,
        help=f"Ignore channels sampled below this rate, in Hz (default: {MIN_SAMPLE_RATE})",
    )

    parser.add_argument(
        "--resample_rate",
        type=float,
        default=RESAMPLE_RATE,
        help=f"Save every channel at this rate, in Hz; 0 keeps the native rate "
             f"(default: {RESAMPLE_RATE:g}, the rate PhaseNet was trained on)",
    )

    parser.add_argument(
        "--min_coverage",
        type=float,
        default=MIN_COVERAGE_FRACTION,
        help=f"Minimum fraction of the window each component must cover (default: {MIN_COVERAGE_FRACTION})",
    )

    parser.add_argument(
        "--max_retries",
        type=int,
        default=MAX_TRANSIENT_RETRIES,
        help=f"Attempts per request after a server/network error (default: {MAX_TRANSIENT_RETRIES})",
    )

    parser.add_argument(
        "--retry_sleep_s",
        type=float,
        default=RETRY_SLEEP_SECONDS,
        help=f"Seconds to wait between attempts (default: {RETRY_SLEEP_SECONDS})",
    )

    parser.add_argument(
        "--fallback_clients",
        default="EARTHSCOPE",
        help="Comma-separated backup data centers tried in order when the "
             "EarthScope federator has no data, e.g. EARTHSCOPE,GEOFON,ETH "
             "(default: EARTHSCOPE)",
    )

    args = parser.parse_args()

    # Command-line options override the module defaults
    TARGET_RATE = float(args.target_rate)
    MIN_SAMPLE_RATE = float(args.min_sample_rate)
    RESAMPLE_RATE = float(args.resample_rate)
    MIN_COVERAGE_FRACTION = float(args.min_coverage)
    MAX_TRANSIENT_RETRIES = int(args.max_retries)
    RETRY_SLEEP_SECONDS = float(args.retry_sleep_s)

    download_waveforms(
        config_json=args.config,
        output_dir=args.output_dir,
        fallback_names=[name.strip() for name in args.fallback_clients.split(",") if name.strip()],
    )