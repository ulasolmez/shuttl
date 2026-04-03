"""
app.py
Streamlit UI for the Shuttle Route Optimizer.

Layout
------
Sidebar  : API key · Destination · Fleet config · Optimize button
Tab 1    : Define Stops (CSV upload or manual form + live table)
Tab 2    : Routes & Map (metrics · per-route details · folium map)
"""

from __future__ import annotations
import os
import io

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from streamlit_folium import st_folium

from maps_client import MapsClient
from optimizer import ShuttleOptimizer, OptimizationResult
from visualizer import build_map

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
load_dotenv()

st.set_page_config(
    page_title="Shuttle Route Optimizer",
    page_icon="🚌",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session-state defaults
# ---------------------------------------------------------------------------
def _init_state():
    defaults = {
        "stops": [],           # list of dicts: name, address, lat, lng, passengers
        "destination": None,   # dict: name, address, lat, lng
        "result": None,        # OptimizationResult | None
        "geo_cache": {},       # address str → (lat, lng)
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_state()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_api_key() -> str:
    """Return API key: sidebar field → st.secrets → .env fallback chain."""
    key = st.session_state.get("api_key_input", "").strip()
    if not key:
        key = st.secrets.get("GOOGLE_MAPS_API_KEY", "")
    if not key:
        key = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
    return key


def _maps_client() -> MapsClient:
    key = _get_api_key()
    if not key:
        raise ValueError(
            "No Google Maps API key found. Enter it in the sidebar or add it to a "
            ".env file as GOOGLE_MAPS_API_KEY."
        )
    return MapsClient(key)


def _geocode_stop(client: MapsClient, address: str) -> tuple[float, float]:
    cache = st.session_state["geo_cache"]
    if address not in cache:
        cache[address] = client.geocode(address)
    return cache[address]


def _stops_dataframe() -> pd.DataFrame:
    stops = st.session_state["stops"]
    if not stops:
        return pd.DataFrame(columns=["#", "Name", "Address", "Lat", "Lng", "Passengers"])
    rows = [
        {
            "#": i + 1,
            "Name": s["name"],
            "Address": s["address"],
            "Lat": round(s["lat"], 6) if s["lat"] is not None else "—",
            "Lng": round(s["lng"], 6) if s["lng"] is not None else "—",
            "Passengers": s["passengers"],
        }
        for i, s in enumerate(stops)
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CSV import helpers
# ---------------------------------------------------------------------------

def _parse_csv(raw: bytes) -> list[dict] | None:
    """
    Parse uploaded CSV. Supports two formats:
      Format A: stop_name, address, passengers
      Format B: stop_name, latitude, longitude, passengers

    Column names are matched case-insensitively; order doesn't matter.
    Returns a list of partial dicts (lat/lng may be None if geocoding is needed).
    """
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        st.error(f"Could not parse CSV: {exc}")
        return None

    df.columns = [c.strip().lower() for c in df.columns]

    # Determine format
    has_latlon = "latitude" in df.columns and "longitude" in df.columns
    has_address = "address" in df.columns

    required_name = next(
        (c for c in df.columns if c in ("stop_name", "name", "stop")),
        None,
    )
    if not required_name:
        st.error("CSV must have a column named 'stop_name', 'name', or 'stop'.")
        return None

    pax_col = next(
        (c for c in df.columns if c in ("passengers", "pax", "count", "headcount")),
        None,
    )
    if not pax_col:
        st.error(
            "CSV must have a column named 'passengers', 'pax', 'count', or 'headcount'."
        )
        return None

    if not has_latlon and not has_address:
        st.error(
            "CSV must contain either an 'address' column or both 'latitude' and "
            "'longitude' columns."
        )
        return None

    stops: list[dict] = []
    for _, row in df.iterrows():
        passengers = int(row[pax_col])
        if passengers <= 0:
            continue

        stop: dict = {
            "name": str(row[required_name]).strip(),
            "passengers": passengers,
            "lat": None,
            "lng": None,
            "address": "",
        }

        if has_latlon:
            stop["lat"] = float(row["latitude"])
            stop["lng"] = float(row["longitude"])
            stop["address"] = stop["name"]  # use name as address label

        if has_address:
            stop["address"] = str(row["address"]).strip()
            if has_latlon:
                pass  # already set above; address used as human label
            # lat/lng will be geocoded later

        stops.append(stop)

    return stops if stops else None


# ---------------------------------------------------------------------------
# Optimization runner
# ---------------------------------------------------------------------------

def _run_optimization(num_vehicles: int, vehicle_capacity: int):
    """Build distance matrix and solve CVRP. Writes result to session state."""
    stops = st.session_state["stops"]
    destination = st.session_state["destination"]

    if not stops:
        st.error("Add at least one pickup stop before optimizing.")
        return
    if not destination:
        st.error("Set a destination in the sidebar before optimizing.")
        return

    try:
        client = _maps_client()
    except ValueError as exc:
        st.error(str(exc))
        return

    with st.spinner("Geocoding stops…"):
        # Geocode any stops that still need lat/lng (address-only CSV rows).
        try:
            for s in stops:
                if s["lat"] is None or s["lng"] is None:
                    s["lat"], s["lng"] = _geocode_stop(client, s["address"])
        except Exception as exc:
            st.error(f"Geocoding failed: {exc}")
            return

    # Build locations list: depot (destination) first, then stops.
    all_locations = [(destination["lat"], destination["lng"])] + [
        (s["lat"], s["lng"]) for s in stops
    ]

    with st.spinner("Fetching driving distances from Google Maps…"):
        try:
            dist_matrix = client.build_distance_matrix(all_locations)
        except Exception as exc:
            st.error(f"Distance matrix request failed: {exc}")
            return

    demands = [0] + [s["passengers"] for s in stops]
    stop_names = [destination["name"]] + [s["name"] for s in stops]

    with st.spinner("Running route optimisation…"):
        try:
            solver = ShuttleOptimizer(
                distance_matrix=dist_matrix,
                demands=demands,
                stop_names=stop_names,
                num_vehicles=num_vehicles,
                vehicle_capacity=vehicle_capacity,
            )
            result = solver.solve()
        except ValueError as exc:
            st.error(str(exc))
            return

    if result.solver_status != "SOLUTION_FOUND":
        st.error(
            "The solver could not find a feasible solution. Try increasing the "
            "number of vehicles or their capacity."
        )
        return

    st.session_state["result"] = result
    st.success("Optimisation complete! Switch to the **Routes & Map** tab.")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🚌 Shuttle Optimizer")
    st.divider()

    # API Key
    st.subheader("Google Maps API Key")
    _secrets_key = st.secrets.get("GOOGLE_MAPS_API_KEY", "")
    _env_key = os.getenv("GOOGLE_MAPS_API_KEY", "")
    if _secrets_key:
        key_hint = "Loaded from Streamlit secrets ✓"
    elif _env_key:
        key_hint = "Loaded from .env ✓"
    else:
        key_hint = "Enter your key below"
    st.caption(key_hint)
    st.text_input(
        "API Key",
        type="password",
        key="api_key_input",
        placeholder="AIza…",
        label_visibility="collapsed",
    )

    st.divider()

    # Destination
    st.subheader("Destination (Hub)")
    dest_address = st.text_input(
        "Destination address",
        placeholder="e.g. 123 Main St, Istanbul",
        key="dest_address_input",
    )
    dest_name = st.text_input(
        "Display name",
        value="Hub",
        key="dest_name_input",
    )

    if st.button("📍 Geocode destination", use_container_width=True):
        if not dest_address.strip():
            st.error("Enter a destination address first.")
        else:
            try:
                client = _maps_client()
                lat, lng = _geocode_stop(client, dest_address.strip())
                st.session_state["destination"] = {
                    "name": dest_name.strip() or "Hub",
                    "address": dest_address.strip(),
                    "lat": lat,
                    "lng": lng,
                }
                st.success(f"Geocoded: {lat:.5f}, {lng:.5f}")
            except Exception as exc:
                st.error(str(exc))

    if st.session_state["destination"]:
        d = st.session_state["destination"]
        st.caption(f"📌 {d['name']} — ({d['lat']:.5f}, {d['lng']:.5f})")

    st.divider()

    # Fleet config
    st.subheader("Fleet Configuration")
    num_vehicles = st.number_input(
        "Number of shuttles available",
        min_value=1,
        max_value=100,
        value=3,
        step=1,
        key="num_vehicles",
    )
    vehicle_capacity = st.number_input(
        "Seats per shuttle",
        min_value=1,
        max_value=200,
        value=15,
        step=1,
        key="vehicle_capacity",
    )

    total_pax = sum(s["passengers"] for s in st.session_state["stops"])
    if total_pax > 0:
        max_cap = num_vehicles * vehicle_capacity
        pct = total_pax / max_cap * 100
        st.caption(
            f"Total passengers: **{total_pax}** | Fleet capacity: **{max_cap}** "
            f"({pct:.0f}% utilised)"
        )

    st.divider()

    optimize_disabled = (
        not st.session_state["stops"] or not st.session_state["destination"]
    )
    if st.button(
        "🔍 Optimise Routes",
        use_container_width=True,
        type="primary",
        disabled=optimize_disabled,
    ):
        _run_optimization(int(num_vehicles), int(vehicle_capacity))

    if optimize_disabled:
        if not st.session_state["destination"]:
            st.caption("⚠ Geocode a destination to enable optimisation.")
        elif not st.session_state["stops"]:
            st.caption("⚠ Add at least one stop to enable optimisation.")


# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------
tab_stops, tab_routes = st.tabs(["📋 Define Stops", "🗺️ Routes & Map"])


# ===========================
# Tab 1 — Define Stops
# ===========================
with tab_stops:
    st.header("Pickup Stops")

    # --- CSV upload ---
    st.subheader("Import from CSV")
    with st.expander("CSV format guide"):
        st.markdown(
            """
**Format A — addresses** (will be geocoded automatically):
| stop_name | address | passengers |
|-----------|---------|------------|
| Stop A    | 45 Istiklal Ave, Istanbul | 8 |

**Format B — coordinates** (no geocoding needed):
| stop_name | latitude | longitude | passengers |
|-----------|----------|-----------|------------|
| Stop A    | 41.0369  | 28.9850   | 8 |

Column names are case-insensitive. Extra columns are ignored.
            """
        )
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded is not None:
        parsed = _parse_csv(uploaded.read())
        if parsed:
            added = 0
            existing_names = {s["name"] for s in st.session_state["stops"]}
            for p in parsed:
                if p["name"] not in existing_names:
                    st.session_state["stops"].append(p)
                    added += 1
            st.success(f"Added {added} stop(s) from CSV ({len(parsed) - added} duplicates skipped).")

    st.divider()

    # --- Manual entry ---
    st.subheader("Add Stop Manually")
    with st.expander("Add a stop"):
        with st.form("add_stop_form", clear_on_submit=True):
            col1, col2 = st.columns([2, 1])
            with col1:
                f_name = st.text_input("Stop name *", placeholder="Office Park A")
                f_address = st.text_input("Address *", placeholder="456 Business Blvd, City")
            with col2:
                f_lat = st.text_input("Latitude (optional)", placeholder="41.0082")
                f_lng = st.text_input("Longitude (optional)", placeholder="28.9784")
                f_pax = st.number_input("Passengers *", min_value=1, value=1, step=1)

            submitted = st.form_submit_button("➕ Add Stop", use_container_width=True)
            if submitted:
                if not f_name.strip():
                    st.error("Stop name is required.")
                elif not f_address.strip():
                    st.error("Address is required.")
                else:
                    lat_val = float(f_lat) if f_lat.strip() else None
                    lng_val = float(f_lng) if f_lng.strip() else None
                    st.session_state["stops"].append({
                        "name": f_name.strip(),
                        "address": f_address.strip(),
                        "lat": lat_val,
                        "lng": lng_val,
                        "passengers": int(f_pax),
                    })
                    st.success(f"Added: {f_name.strip()}")

    st.divider()

    # --- Current stops table ---
    st.subheader(f"Current Stops ({len(st.session_state['stops'])})")

    if st.session_state["stops"]:
        df = _stops_dataframe()
        st.dataframe(df, use_container_width=True, hide_index=True)

        c1, c2 = st.columns([1, 4])
        with c1:
            if st.button("🗑️ Clear all stops", type="secondary"):
                st.session_state["stops"] = []
                st.session_state["result"] = None
                st.rerun()
        with c2:
            # Download current stops as CSV
            csv_bytes = df.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download stops CSV",
                data=csv_bytes,
                file_name="stops.csv",
                mime="text/csv",
            )

        # Remove individual stop
        with st.expander("Remove a specific stop"):
            stop_names_list = [s["name"] for s in st.session_state["stops"]]
            to_remove = st.selectbox("Select stop to remove", stop_names_list)
            if st.button("Remove selected stop"):
                st.session_state["stops"] = [
                    s for s in st.session_state["stops"] if s["name"] != to_remove
                ]
                st.session_state["result"] = None
                st.rerun()
    else:
        st.info("No stops defined yet. Upload a CSV or add stops manually above.")


# ===========================
# Tab 2 — Routes & Map
# ===========================
with tab_routes:
    st.header("Optimised Routes")

    result: OptimizationResult | None = st.session_state["result"]

    if result is None:
        st.info(
            "No results yet. Define your stops and destination, then click "
            "**Optimise Routes** in the sidebar."
        )
    else:
        stops = st.session_state["stops"]
        destination = st.session_state["destination"]

        # --- Summary metrics ---
        total_dist_km = round(result.total_distance_m / 1000, 1)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Shuttles Used", result.vehicles_used)
        m2.metric("Total Passengers", result.total_passengers)
        m3.metric("Avg Occupancy", f"{result.avg_occupancy_pct}%")
        m4.metric("Total Distance", f"{total_dist_km} km")

        st.divider()

        # --- Per-route details ---
        st.subheader("Route Details")
        for route in result.routes:
            color_dot = "🔴🔵🟢🟠🟣🔵🟤"[route.vehicle_id % 7]
            with st.expander(
                f"{color_dot} Shuttle {route.vehicle_id + 1} — "
                f"{route.total_passengers} passengers · "
                f"{route.occupancy_pct}% occupancy · "
                f"{round(route.total_distance_m / 1000, 1)} km"
            ):
                # Build stop sequence table (skip depot at start/end).
                route_rows = []
                for seq, (node_idx, node_name) in enumerate(
                    zip(route.stop_indices, route.stop_names)
                ):
                    if node_idx == 0:
                        label = "🏢 Hub (Destination)"
                        pax = "—"
                    else:
                        s = stops[node_idx - 1]
                        label = node_name
                        pax = s["passengers"]
                    route_rows.append({"Stop": seq + 1, "Name": label, "Passengers": pax})

                st.dataframe(
                    pd.DataFrame(route_rows),
                    use_container_width=True,
                    hide_index=True,
                )

        st.divider()

        # --- Folium map ---
        st.subheader("Route Map")

        try:
            fmap = build_map(stops, destination, result)
            st_folium(fmap, use_container_width=True, height=580, returned_objects=[])
        except Exception as exc:
            st.error(f"Map rendering failed: {exc}")

        st.divider()

        # --- Download route plan ---
        route_rows_all = []
        for route in result.routes:
            for seq, (node_idx, node_name) in enumerate(
                zip(route.stop_indices, route.stop_names)
            ):
                if node_idx == 0:
                    pax = 0
                    address = destination.get("address", "")
                else:
                    s = stops[node_idx - 1]
                    pax = s["passengers"]
                    address = s.get("address", "")
                route_rows_all.append({
                    "Shuttle": f"Shuttle {route.vehicle_id + 1}",
                    "Sequence": seq + 1,
                    "Stop Name": node_name,
                    "Address": address,
                    "Passengers": pax,
                    "Occupancy %": route.occupancy_pct,
                    "Distance (km)": round(route.total_distance_m / 1000, 1),
                })

        csv_result = pd.DataFrame(route_rows_all).to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download route plan as CSV",
            data=csv_result,
            file_name="shuttle_routes.csv",
            mime="text/csv",
        )
