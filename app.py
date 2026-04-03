"""
app.py
Streamlit UI for the Shuttle Route Optimizer.

Layout
------
Sidebar  : Map mode · API key · Fleet config · Optimize button
Main     : Interactive Folium map (always visible) → click to add nodes
           Pending-click confirmation form
           Stop list + CSV import
           Route results (post-optimisation)
"""

from __future__ import annotations
import os
import io
import json

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
        "stops": [],                # list of {name, address, lat, lng, passengers, occupations?}
        "destination": None,        # {name, address, lat, lng} | None
        "result": None,             # OptimizationResult | None
        "route_polylines": None,    # {(from_node, to_node): [(lat, lng), …]} | None
        "geo_cache": {},            # address → (lat, lng) cache
        "pending_click": None,      # {lat, lng} awaiting user confirmation
        "_last_click_coords": None, # dedup guard: last processed (lat, lng) tuple
        "occupation_types": ["Worker", "Technician", "Cleaner"],
        "occ_optimize": False,      # group same occupations onto same shuttle when possible
        "snap_notice": None,        # name of last stop that was auto-snapped to road
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_state()


# ---------------------------------------------------------------------------
# API / client helpers
# ---------------------------------------------------------------------------

def _get_api_key() -> str:
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
            "No Google Maps API key found. Enter it in the sidebar or set "
            "GOOGLE_MAPS_API_KEY in .env / Streamlit secrets."
        )
    return MapsClient(key)


def _geocode_cached(client: MapsClient, address: str) -> tuple[float, float]:
    cache = st.session_state["geo_cache"]
    if address not in cache:
        cache[address] = client.geocode(address)
    return cache[address]


# ---------------------------------------------------------------------------
# Stop table helper
# ---------------------------------------------------------------------------

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
            "Occupations": ", ".join(
                f"{k}×{v}" for k, v in s.get("occupations", {}).items() if v > 0
            ) or "—",
        }
        for i, s in enumerate(stops)
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Session save / load (JSON)
# ---------------------------------------------------------------------------

def _session_to_json() -> bytes:
    """Serialise stops + hub to JSON bytes for download."""
    payload = {
        "version": 1,
        "destination": st.session_state["destination"],
        "stops": st.session_state["stops"],
        "occupation_types": st.session_state["occupation_types"],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode()


def _load_session_json(raw: bytes) -> str | None:
    """
    Deserialise a previously exported session JSON.
    Returns an error string on failure, None on success.
    """
    try:
        payload = json.loads(raw.decode())
    except Exception:
        return "File is not valid JSON."

    if not isinstance(payload, dict) or payload.get("version") != 1:
        return "Unrecognised file format (expected version 1 session file)."

    dest = payload.get("destination")
    stops = payload.get("stops", [])

    if dest is not None:
        required_dest = {"name", "address", "lat", "lng"}
        if not required_dest.issubset(dest.keys()):
            return "Destination entry is missing required fields."

    for s in stops:
        if not {"name", "address", "passengers"}.issubset(s.keys()):
            return "One or more stop entries are missing required fields."

    st.session_state["destination"] = dest
    st.session_state["stops"] = stops
    if "occupation_types" in payload and isinstance(payload["occupation_types"], list):
        st.session_state["occupation_types"] = payload["occupation_types"]
    st.session_state["result"] = None
    st.session_state["route_polylines"] = None
    st.session_state["pending_click"] = None
    st.session_state["_last_click_coords"] = None
    return None


# ---------------------------------------------------------------------------
# CSV import
# ---------------------------------------------------------------------------

def _parse_csv(raw: bytes) -> list[dict] | None:
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        st.error(f"Could not parse CSV: {exc}")
        return None

    df.columns = [c.strip().lower() for c in df.columns]
    has_latlon = "latitude" in df.columns and "longitude" in df.columns
    has_address = "address" in df.columns

    name_col = next((c for c in df.columns if c in ("stop_name", "name", "stop")), None)
    pax_col = next((c for c in df.columns if c in ("passengers", "pax", "count", "headcount")), None)

    if not name_col:
        st.error("CSV must have a column named 'stop_name', 'name', or 'stop'.")
        return None
    if not pax_col:
        st.error("CSV must have a column named 'passengers', 'pax', 'count', or 'headcount'.")
        return None
    if not has_latlon and not has_address:
        st.error("CSV must have an 'address' column or both 'latitude' and 'longitude' columns.")
        return None

    stops: list[dict] = []
    for _, row in df.iterrows():
        # Skip rows where passengers or name is missing/NaN.
        if pd.isna(row[pax_col]) or pd.isna(row[name_col]):
            continue
        try:
            pax = int(float(row[pax_col]))
        except (ValueError, TypeError):
            continue
        if pax <= 0:
            continue
        stop: dict = {"name": str(row[name_col]).strip(), "passengers": pax,
                      "lat": None, "lng": None, "address": ""}
        if has_latlon:
            stop["lat"] = float(row["latitude"])
            stop["lng"] = float(row["longitude"])
            stop["address"] = stop["name"]
        if has_address:
            stop["address"] = str(row["address"]).strip()
        stops.append(stop)

    return stops or None


# ---------------------------------------------------------------------------
# Optimisation runner
# ---------------------------------------------------------------------------

def _run_optimization(num_vehicles: int, vehicle_capacity: int, max_distance_km: float,
                      occ_optimize: bool = False):
    stops = st.session_state["stops"]
    destination = st.session_state["destination"]

    if not stops:
        st.error("Add at least one pickup stop before optimising.")
        return
    if not destination:
        st.error("Set a hub destination before optimising.")
        return

    try:
        client = _maps_client()
    except ValueError as exc:
        st.error(str(exc))
        return

    # Geocode any address-only stops.
    with st.spinner("Geocoding stops…"):
        try:
            for s in stops:
                if s["lat"] is None or s["lng"] is None:
                    s["lat"], s["lng"] = _geocode_cached(client, s["address"])
        except Exception as exc:
            st.error(f"Geocoding failed: {exc}")
            return

    all_locations = [(destination["lat"], destination["lng"])] + [
        (s["lat"], s["lng"]) for s in stops
    ]

    with st.spinner("Fetching driving distances (Distance Matrix API)…"):
        try:
            dist_matrix = client.build_distance_matrix(all_locations)
        except Exception as exc:
            st.error(f"Distance matrix request failed: {exc}")
            return

    # Detect stops that cannot reach the hub.  A stop is unreachable when its
    # arc to node 0 (depot/hub) is the penalty value used for ZERO_RESULTS cells.
    _PENALTY = 999_999_999
    _unreachable = [
        stops[_si]["name"]
        for _si in range(len(stops))
        if dist_matrix[_si + 1][0] >= _PENALTY or dist_matrix[0][_si + 1] >= _PENALTY
    ]
    if _unreachable:
        _names = ", ".join(f"**{n}**" for n in _unreachable)
        st.error(
            f"The following stop(s) have no driveable route to the hub: {_names}. "
            "Remove them and re-add by clicking on a nearby road."
        )
        return

    # Occupation-aware routing: penalise arcs between stops of different primary types.
    # Penalty per mixed edge < fixed vehicle cost → solver prefers grouping but won't
    # open an extra shuttle just to keep occupations separate.
    if occ_optimize and any(s.get("occupations") for s in stops):
        _max_arc = max((max(row) for row in dist_matrix), default=1)
        _penalty = max(1, _max_arc // max(1, len(stops)))

        def _primary_occ(s: dict) -> str | None:
            occs = s.get("occupations", {})
            return max(occs, key=lambda k: occs[k]) if occs else None

        _penalized = [row[:] for row in dist_matrix]
        for _pi_idx in range(1, len(dist_matrix)):
            _pi = _primary_occ(stops[_pi_idx - 1])
            if _pi is None:
                continue
            for _pj_idx in range(1, len(dist_matrix)):
                _pj = _primary_occ(stops[_pj_idx - 1])
                if _pj is not None and _pi != _pj:
                    _penalized[_pi_idx][_pj_idx] += _penalty
        dist_matrix = _penalized

    demands = [0] + [s["passengers"] for s in stops]
    stop_names = [destination["name"]] + [s["name"] for s in stops]

    with st.spinner("Running route optimisation (OR-Tools CVRP)…"):
        try:
            solver = ShuttleOptimizer(
                distance_matrix=dist_matrix,
                demands=demands,
                stop_names=stop_names,
                num_vehicles=num_vehicles,
                vehicle_capacity=vehicle_capacity,
                max_route_distance_m=int(max_distance_km * 1000),
            )
            result = solver.solve()
        except ValueError as exc:
            st.error(str(exc))
            return

    if result.solver_status != "SOLUTION_FOUND":
        st.error(
            "The solver could not find a feasible solution. "
            "Try adding more vehicles or increasing capacity."
        )
        return

    # Fetch street-following polylines for every edge in the solution.
    route_polylines: dict[tuple[int, int], list[tuple[float, float]]] = {}
    total_edges = sum(
        len(r.stop_indices) - 1 for r in result.routes
    )
    with st.spinner(f"Fetching {total_edges} street route segments (Directions API)…"):
        for route in result.routes:
            nodes = route.stop_indices
            for i in range(len(nodes) - 1):
                fn, tn = nodes[i], nodes[i + 1]
                key = (fn, tn)
                if key not in route_polylines:
                    origin = all_locations[fn] if fn < len(all_locations) else all_locations[0]
                    dest = all_locations[tn] if tn < len(all_locations) else all_locations[0]
                    route_polylines[key] = client.get_route_polyline(origin, dest)

    st.session_state["result"] = result
    st.session_state["route_polylines"] = route_polylines
    st.rerun()


# ===========================================================================
# SIDEBAR
# ===========================================================================

with st.sidebar:
    st.title("🚌 Shuttle Optimizer")
    st.divider()

    # Map interaction mode
    st.subheader("Map Mode")
    map_mode = st.radio(
        "Click the map to:",
        options=["🚏 Add Stop", "🏢 Set Hub"],
        key="map_mode",
        horizontal=True,
        label_visibility="collapsed",
    )
    if map_mode == "🚏 Add Stop":
        st.caption("Click anywhere on the map to place a new pickup stop.")
    else:
        st.caption("Click anywhere on the map to place / move the hub.")

    st.divider()

    # Fleet
    st.subheader("Fleet Configuration")
    num_vehicles = st.number_input("Shuttles available", min_value=1,
                                   max_value=100, value=3, step=1, key="num_vehicles")
    vehicle_capacity = st.number_input("Seats per shuttle", min_value=1,
                                       max_value=200, value=15, step=1, key="vehicle_capacity")
    max_distance_km = st.number_input("Max route distance (km)", min_value=1,
                                      max_value=1000, value=30, step=5,
                                      key="max_distance_km",
                                      help="Maximum total driving distance per shuttle route.")

    total_pax = sum(s["passengers"] for s in st.session_state["stops"])
    if total_pax > 0:
        max_cap = int(num_vehicles * vehicle_capacity)
        pct = total_pax / max_cap * 100
        st.caption(f"Passengers: **{total_pax}** / Capacity: **{max_cap}** ({pct:.0f}%)")

    st.divider()

    # Occupation types manager
    st.subheader("Occupation Types")
    for _oi, _ot in enumerate(st.session_state["occupation_types"]):
        _c1, _c2 = st.columns([5, 1])
        with _c1:
            st.text(_ot)
        with _c2:
            if st.button("✕", key=f"rm_occ_{_oi}", help=f"Remove {_ot}"):
                st.session_state["occupation_types"].pop(_oi)
                st.rerun()
    _nc1, _nc2 = st.columns([3, 1])
    with _nc1:
        st.text_input("Add type", key="new_occ_type",
                      label_visibility="collapsed", placeholder="e.g. Engineer")
    with _nc2:
        if st.button("Add", key="add_occ_btn"):
            _nval = st.session_state.get("new_occ_type", "").strip()
            if _nval and _nval not in st.session_state["occupation_types"]:
                st.session_state["occupation_types"].append(_nval)
                st.rerun()
    st.checkbox(
        "Optimize by occupation",
        key="occ_optimize",
        help="Penalises mixed-occupation routes. Shuttles still fill spare seats "
             "with other roles when needed (no extra vehicle is added just to separate).",
    )

    st.divider()

    # Optimize button
    _can_optimize = bool(st.session_state["stops"]) and bool(st.session_state["destination"])
    if st.button("🔍 Optimise Routes", use_container_width=True,
                 type="primary", disabled=not _can_optimize):
        _run_optimization(int(num_vehicles), int(vehicle_capacity), float(max_distance_km),
                          bool(st.session_state.get("occ_optimize", False)))

    if not _can_optimize:
        if not st.session_state["destination"]:
            st.caption("⚠ Set a hub first (click map in Set Hub mode).")
        elif not st.session_state["stops"]:
            st.caption("⚠ Add at least one stop.")

    # Clear all
    if st.session_state["stops"] or st.session_state["destination"]:
        st.divider()
        if st.button("🗑️ Clear everything", use_container_width=True, type="secondary"):
            st.session_state.update({
                "stops": [], "destination": None,
                "result": None, "route_polylines": None,
                "pending_click": None, "_last_click_coords": None,
                "snap_notice": None,
            })
            st.rerun()


# ===========================================================================
# MAIN AREA
# ===========================================================================

st.title("🚌 Shuttle Route Optimizer")

# --- Build and render the map ---
fmap = build_map(
    stops=st.session_state["stops"],
    destination=st.session_state["destination"],
    result=st.session_state["result"],
    route_polylines=st.session_state["route_polylines"],
    pending_click=st.session_state["pending_click"],
)

map_data = st_folium(
    fmap,
    key="main_map",
    use_container_width=True,
    height=520,
    returned_objects=["last_clicked"],
)

# ---------------------------------------------------------------------------
# Click detection — only when no pending form is open
# ---------------------------------------------------------------------------
if st.session_state["pending_click"] is None:
    click = map_data.get("last_clicked") if map_data else None
    if click:
        new_coords = (round(click["lat"], 7), round(click["lng"], 7))
        if new_coords != st.session_state["_last_click_coords"]:
            st.session_state["_last_click_coords"] = new_coords
            st.session_state["pending_click"] = {
                "lat": click["lat"],
                "lng": click["lng"],
                "mode": st.session_state["map_mode"],
            }
            st.rerun()

# ---------------------------------------------------------------------------
# Pending-click confirmation form
# ---------------------------------------------------------------------------
pc = st.session_state["pending_click"]
if pc:
    lat_str = f"{pc['lat']:.6f}"
    lng_str = f"{pc['lng']:.6f}"
    mode = pc.get("mode", "🚏 Add Stop")

    if mode == "🏢 Set Hub":
        st.info(f"📍 Setting hub at ({lat_str}, {lng_str})")
        with st.form("hub_form"):
            hub_name = st.text_input("Hub name", value="Hub")
            c1, c2 = st.columns(2)
            with c1:
                if st.form_submit_button("✅ Set as Hub", use_container_width=True):
                    _h_lat, _h_lng, _h_addr = pc["lat"], pc["lng"], f"{lat_str},{lng_str}"
                    try:
                        _mc = _maps_client()
                        _h_lat, _h_lng, _h_addr_snap, _ = _mc.snap_to_road(pc["lat"], pc["lng"])
                        if _h_addr_snap:
                            _h_addr = _h_addr_snap
                    except Exception:
                        pass
                    st.session_state["destination"] = {
                        "name": hub_name.strip() or "Hub",
                        "address": _h_addr,
                        "lat": _h_lat,
                        "lng": _h_lng,
                    }
                    # Moving hub invalidates previous optimisation result.
                    st.session_state["result"] = None
                    st.session_state["route_polylines"] = None
                    st.session_state["pending_click"] = None
                    st.rerun()
            with c2:
                if st.form_submit_button("✗ Cancel", use_container_width=True):
                    st.session_state["pending_click"] = None
                    st.rerun()
    else:
        st.info(f"📍 New stop at ({lat_str}, {lng_str})")
        _occ_types = st.session_state["occupation_types"]
        with st.form("stop_form"):
            stop_name = st.text_input("Stop name *", placeholder="Bus Stop A")
            if _occ_types:
                st.markdown("**Occupation breakdown** — leave all zero to enter a total instead")
                _col_count = min(len(_occ_types), 4)
                _occ_cols = st.columns(_col_count)
                _occ_vals: dict[str, int] = {}
                for _oi, _ot in enumerate(_occ_types):
                    with _occ_cols[_oi % _col_count]:
                        _occ_vals[_ot] = st.number_input(_ot, min_value=0, value=0, step=1,
                                                         key=f"new_stop_occ_{_ot}")
                stop_pax = st.number_input(
                    "Total passengers (used only if all above are zero)",
                    min_value=1, value=1, step=1, key="new_stop_pax_fallback")
            else:
                _occ_vals = {}
                stop_pax = st.number_input("Passengers *", min_value=1, value=1, step=1)
            c1, c2 = st.columns(2)
            with c1:
                if st.form_submit_button("✅ Add Stop", use_container_width=True):
                    if not stop_name.strip():
                        st.error("Name is required.")
                    else:
                        _occ_total = sum(_occ_vals.values())
                        passengers = _occ_total if _occ_total > 0 else int(stop_pax)
                        # Snap to road via Directions API (uses hub as reference when
                        # available — same API as optimizer, reliable snap + reachability).
                        _s_lat, _s_lng = pc["lat"], pc["lng"]
                        _s_addr = f"{lat_str},{lng_str}"
                        _snapped = False
                        _hub = st.session_state.get("destination")
                        try:
                            _mc = _maps_client()
                            _ref_lat = _hub["lat"] if _hub else None
                            _ref_lng = _hub["lng"] if _hub else None
                            _sn_lat, _sn_lng, _sn_addr, _reachable = _mc.snap_to_road(
                                pc["lat"], pc["lng"], _ref_lat, _ref_lng
                            )
                            if not _reachable:
                                st.error(
                                    "🚫 This location has no driveable road access. "
                                    "Please click on or near a road."
                                )
                                st.stop()
                            _dist_m = (((_sn_lat - pc["lat"]) * 111_000) ** 2
                                       + ((_sn_lng - pc["lng"]) * 111_000) ** 2) ** 0.5
                            _s_lat, _s_lng = _sn_lat, _sn_lng
                            if _sn_addr:
                                _s_addr = _sn_addr
                            _snapped = _dist_m > 5
                        except ValueError:
                            # No API key yet — skip snap but allow add.
                            pass
                        _new_stop: dict = {
                            "name": stop_name.strip(),
                            "address": _s_addr,
                            "lat": _s_lat,
                            "lng": _s_lng,
                            "passengers": passengers,
                        }
                        if _occ_total > 0:
                            _new_stop["occupations"] = {k: v for k, v in _occ_vals.items() if v > 0}
                        if _snapped:
                            _new_stop["snapped"] = True
                        st.session_state["stops"].append(_new_stop)
                        if _snapped:
                            st.session_state["snap_notice"] = stop_name.strip()
                        # Invalidate previous result when graph changes.
                        st.session_state["result"] = None
                        st.session_state["route_polylines"] = None
                        st.session_state["pending_click"] = None
                        st.rerun()
            with c2:
                if st.form_submit_button("✗ Cancel", use_container_width=True):
                    st.session_state["pending_click"] = None
                    st.rerun()

st.divider()

# Show snap notice once after a stop is snapped to road.
if st.session_state.get("snap_notice"):
    st.info(
        f"📌 '{st.session_state['snap_notice']}' was moved to the nearest "
        "driveable road automatically."
    )
    st.session_state["snap_notice"] = None

# ---------------------------------------------------------------------------
# Stop list + CSV import (collapsible)
# ---------------------------------------------------------------------------
with st.expander(
    f"📋 Pickup Stops ({len(st.session_state['stops'])})"
    + (" — click to manage" if st.session_state["stops"] else " — none yet"),
    expanded=not st.session_state["stops"],
):
    # Session save / load
    with st.expander("💾 Save / Load Session"):
        has_data = bool(st.session_state["stops"]) or bool(st.session_state["destination"])
        st.caption("Export your stops and hub to a file so you can reload them later.")
        col_save, col_load = st.columns(2)
        with col_save:
            st.download_button(
                "⬇️ Export session (.json)",
                data=_session_to_json(),
                file_name="shuttl_session.json",
                mime="application/json",
                disabled=not has_data,
                use_container_width=True,
            )
        with col_load:
            session_file = st.file_uploader(
                "Import session (.json)", type=["json"], key="session_upload",
                label_visibility="collapsed",
            )
            if session_file is not None:
                err = _load_session_json(session_file.read())
                if err:
                    st.error(err)
                else:
                    st.success("Session loaded!")
                    st.rerun()

    # CSV upload
    with st.expander("⬆️ Import from CSV"):
        st.markdown(
            "**Format A** — `stop_name, address, passengers`  \n"
            "**Format B** — `stop_name, latitude, longitude, passengers`"
        )
        uploaded = st.file_uploader("Upload CSV", type=["csv"], key="csv_upload")
        if uploaded is not None:
            parsed = _parse_csv(uploaded.read())
            if parsed:
                existing = {s["name"] for s in st.session_state["stops"]}
                added = 0
                for p in parsed:
                    if p["name"] not in existing:
                        st.session_state["stops"].append(p)
                        added += 1
                st.success(f"Added {added} stop(s).")
                st.session_state["result"] = None
                st.session_state["route_polylines"] = None
                st.rerun()

    if st.session_state["stops"]:
        df = _stops_dataframe()
        st.dataframe(df, use_container_width=True, hide_index=True)

        col_dl, col_rm = st.columns([1, 2])
        with col_dl:
            st.download_button(
                "⬇️ Download CSV",
                data=df.to_csv(index=False).encode(),
                file_name="stops.csv",
                mime="text/csv",
            )
        with col_rm:
            with st.expander("Remove a stop"):
                to_remove = st.selectbox(
                    "Stop", [s["name"] for s in st.session_state["stops"]], key="rm_sel"
                )
                if st.button("Remove", key="rm_btn"):
                    st.session_state["stops"] = [
                        s for s in st.session_state["stops"] if s["name"] != to_remove
                    ]
                    st.session_state["result"] = None
                    st.session_state["route_polylines"] = None
                    st.rerun()

        with st.expander("✏️ Edit stop occupations"):
            _eocc_types = st.session_state["occupation_types"]
            if not _eocc_types:
                st.caption("Define occupation types in the sidebar first.")
            else:
                _edit_name = st.selectbox(
                    "Select stop",
                    [s["name"] for s in st.session_state["stops"]],
                    key="edit_occ_sel",
                )
                _edit_idx = next(
                    (i for i, s in enumerate(st.session_state["stops"])
                     if s["name"] == _edit_name),
                    None,
                )
                if _edit_idx is not None:
                    _s = st.session_state["stops"][_edit_idx]
                    _cur_occs = _s.get("occupations", {})
                    with st.form("edit_occ_form"):
                        st.caption(f"Current total: **{_s['passengers']}** passengers")
                        _ec = min(len(_eocc_types), 4)
                        _ecols = st.columns(_ec)
                        _new_occ_vals: dict[str, int] = {}
                        for _oi2, _ot2 in enumerate(_eocc_types):
                            with _ecols[_oi2 % _ec]:
                                _new_occ_vals[_ot2] = st.number_input(
                                    _ot2, min_value=0, step=1,
                                    value=int(_cur_occs.get(_ot2, 0)),
                                    key=f"edit_occ_{_ot2}",
                                )
                        _pax_fb = st.number_input(
                            "Total passengers (if no occupation breakdown)",
                            min_value=1, value=_s["passengers"], step=1,
                            key="edit_pax_fallback",
                        )
                        if st.form_submit_button("💾 Save changes", use_container_width=True):
                            _occ_sum = sum(_new_occ_vals.values())
                            if _occ_sum > 0:
                                st.session_state["stops"][_edit_idx]["occupations"] = {
                                    k: v for k, v in _new_occ_vals.items() if v > 0
                                }
                                st.session_state["stops"][_edit_idx]["passengers"] = _occ_sum
                            else:
                                st.session_state["stops"][_edit_idx].pop("occupations", None)
                                st.session_state["stops"][_edit_idx]["passengers"] = int(_pax_fb)
                            st.session_state["result"] = None
                            st.session_state["route_polylines"] = None
                            st.rerun()
    else:
        st.info("No stops yet. Click the map (Add Stop mode) or upload a CSV.")

# ---------------------------------------------------------------------------
# Route results (post-optimisation)
# ---------------------------------------------------------------------------
result: OptimizationResult | None = st.session_state["result"]

if result is not None:
    st.divider()
    st.subheader("📊 Optimisation Results")

    total_dist_km = round(result.total_distance_m / 1000, 1)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Shuttles Used", result.vehicles_used)
    m2.metric("Total Passengers", result.total_passengers)
    m3.metric("Avg Occupancy", f"{result.avg_occupancy_pct}%")
    m4.metric("Total Distance", f"{total_dist_km} km")

    st.divider()

    stops_list = st.session_state["stops"]
    destination = st.session_state["destination"]

    for route in result.routes:
        emoji = "🔴🔵🟢🟠🟣🔵🟤"[route.vehicle_id % 7]
        # Build occupation summary for this shuttle's route.
        _route_occs: dict[str, int] = {}
        for _rn in route.stop_indices:
            if _rn != 0:
                for _occ, _cnt in stops_list[_rn - 1].get("occupations", {}).items():
                    _route_occs[_occ] = _route_occs.get(_occ, 0) + _cnt
        _occ_summary = ", ".join(f"{k}: {v}" for k, v in sorted(_route_occs.items()))
        with st.expander(
            f"{emoji} Shuttle {route.vehicle_id + 1} — "
            f"{route.total_passengers} pax · "
            f"{route.occupancy_pct}% occupancy · "
            f"{round(route.total_distance_m / 1000, 1)} km"
            + (f" · {_occ_summary}" if _occ_summary else "")
        ):
            rows = []
            for seq, (node_idx, node_name) in enumerate(
                zip(route.stop_indices, route.stop_names)
            ):
                if node_idx == 0:
                    label, pax, occ_str = "🏢 Hub (Destination)", "—", "—"
                else:
                    s = stops_list[node_idx - 1]
                    label, pax = node_name, s["passengers"]
                    occ_str = ", ".join(
                        f"{k}×{v}" for k, v in s.get("occupations", {}).items() if v > 0
                    ) or "—"
                rows.append({"Stop #": seq + 1, "Name": label, "Passengers": pax,
                             "Occupations": occ_str})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Download
    all_rows = []
    for route in result.routes:
        for seq, (node_idx, node_name) in enumerate(
            zip(route.stop_indices, route.stop_names)
        ):
            if node_idx == 0:
                pax, addr = 0, destination.get("address", "")
            else:
                s = stops_list[node_idx - 1]
                pax, addr = s["passengers"], s.get("address", "")
            all_rows.append({
                "Shuttle": f"Shuttle {route.vehicle_id + 1}",
                "Sequence": seq + 1,
                "Stop Name": node_name,
                "Address": addr,
                "Passengers": pax,
                "Occupancy %": route.occupancy_pct,
                "Distance (km)": round(route.total_distance_m / 1000, 1),
            })
    st.download_button(
        "⬇️ Download route plan CSV",
        data=pd.DataFrame(all_rows).to_csv(index=False).encode(),
        file_name="shuttle_routes.csv",
        mime="text/csv",
    )

