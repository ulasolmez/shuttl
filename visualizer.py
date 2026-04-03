"""
visualizer.py
Builds a Folium map showing shuttle routes, stop markers, and the hub.
"""

from __future__ import annotations
import folium
from optimizer import OptimizationResult

# ---------------------------------------------------------------------------
# Colour palette — distinct enough to tell routes apart on a map
# ---------------------------------------------------------------------------
_ROUTE_COLORS = [
    "#E63946",  # red
    "#2196F3",  # blue
    "#4CAF50",  # green
    "#FF9800",  # orange
    "#9C27B0",  # purple
    "#00BCD4",  # cyan
    "#FF5722",  # deep orange
    "#8BC34A",  # light green
    "#3F51B5",  # indigo
    "#F06292",  # pink
    "#795548",  # brown
    "#607D8B",  # blue-grey
]


def _route_color(vehicle_id: int) -> str:
    return _ROUTE_COLORS[vehicle_id % len(_ROUTE_COLORS)]


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def build_map(
    stops: list[dict],          # list of stop dicts with keys: name, lat, lng, passengers
    destination: dict,          # dict with keys: name, lat, lng
    result: OptimizationResult,
) -> folium.Map:
    """
    Render a Folium map with:
    - A hub/destination marker (star icon)
    - Coloured circle markers for each pickup stop
    - PolyLines showing each shuttle's route (from depot through stops and back)

    Parameters
    ----------
    stops       : all pickup stops (index-aligned so index 0 in stops corresponds
                  to node 1 in the distance matrix, i.e. depot is excluded).
    destination : the single hub / destination.
    result      : OptimizationResult from the solver.

    Returns
    -------
    folium.Map ready to be rendered by streamlit-folium.
    """
    # Centre map on the destination.
    fmap = folium.Map(
        location=[destination["lat"], destination["lng"]],
        zoom_start=12,
        tiles="OpenStreetMap",
    )

    # --- Hub marker ---
    folium.Marker(
        location=[destination["lat"], destination["lng"]],
        popup=folium.Popup(
            f"<b>🏢 {destination['name']}</b><br><i>Hub / Destination</i>",
            max_width=200,
        ),
        tooltip=destination["name"],
        icon=folium.Icon(color="black", icon="home", prefix="fa"),
    ).add_to(fmap)

    # Build a quick lookup: stop index (1-based in distance matrix) → stop dict
    # stops list is 0-based and does NOT include the depot, so stops[i] = node i+1
    stop_lookup: dict[int, dict] = {i + 1: s for i, s in enumerate(stops)}

    # --- Route lines + stop markers ---
    for route in result.routes:
        color = _route_color(route.vehicle_id)

        # Collect ordered lat/lng pairs for the polyline.
        # route.stop_indices is: [0 (depot), stop_node, …, 0 (depot)]
        coords: list[tuple[float, float]] = []
        for node in route.stop_indices:
            if node == 0:
                # depot
                coords.append((destination["lat"], destination["lng"]))
            else:
                s = stop_lookup.get(node)
                if s:
                    coords.append((s["lat"], s["lng"]))

        # Draw route line.
        if len(coords) >= 2:
            folium.PolyLine(
                locations=coords,
                color=color,
                weight=4,
                opacity=0.85,
                tooltip=f"Shuttle {route.vehicle_id + 1} — {route.total_passengers} pax"
                        f" ({route.occupancy_pct}% full)",
            ).add_to(fmap)

        # Draw stop markers (skip depot nodes).
        for seq_idx, node in enumerate(route.stop_indices):
            if node == 0:
                continue
            s = stop_lookup.get(node)
            if not s:
                continue

            popup_html = (
                f"<b>{s['name']}</b><br>"
                f"Passengers: {s['passengers']}<br>"
                f"Shuttle {route.vehicle_id + 1} — stop #{seq_idx}<br>"
                f"Occupancy: {route.occupancy_pct}%"
            )
            folium.CircleMarker(
                location=[s["lat"], s["lng"]],
                radius=10,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.85,
                popup=folium.Popup(popup_html, max_width=220),
                tooltip=f"{s['name']} ({s['passengers']} pax)",
            ).add_to(fmap)

    # Add a legend as a custom HTML control.
    legend_items = "".join(
        f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">'
        f'<div style="width:16px;height:16px;border-radius:50%;background:{_route_color(r.vehicle_id)};"></div>'
        f'<span>Shuttle {r.vehicle_id + 1} — {r.total_passengers} pax ({r.occupancy_pct}%)</span>'
        f'</div>'
        for r in result.routes
    )
    legend_html = f"""
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
                padding:12px 16px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.25);
                font-family:sans-serif;font-size:13px;max-width:240px;">
      <b style="font-size:14px;">Route Legend</b>
      <div style="margin-top:8px;">{legend_items}</div>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))

    return fmap
