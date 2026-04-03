"""
visualizer.py
Builds a Folium map showing the graph of stops as nodes and shuttle routes
as edges rendered along actual streets (via pre-computed polylines).

The map is rendered at all times — before and after optimisation.
"""

from __future__ import annotations
import folium
from optimizer import OptimizationResult

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_ROUTE_COLORS = [
    "#E63946", "#2196F3", "#4CAF50", "#FF9800", "#9C27B0",
    "#00BCD4", "#FF5722", "#8BC34A", "#3F51B5", "#F06292",
    "#795548", "#607D8B",
]

_UNASSIGNED_COLOR = "#9E9E9E"


def _route_color(vehicle_id: int) -> str:
    return _ROUTE_COLORS[vehicle_id % len(_ROUTE_COLORS)]


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def build_map(
    stops: list[dict],
    destination: dict | None = None,
    result: OptimizationResult | None = None,
    route_polylines: dict[tuple[int, int], list[tuple[float, float]]] | None = None,
    pending_click: dict | None = None,
) -> folium.Map:
    """
    Render a Folium map.

    Pre-optimisation  : shows stop markers (grey) + hub marker, no edges.
    Post-optimisation : adds coloured route edges following real streets
                        and colours stop markers by shuttle assignment.

    Parameters
    ----------
    stops           : pickup stop dicts {name, lat, lng, passengers}.
    destination     : hub dict {name, lat, lng}. May be None before it is set.
    result          : OptimizationResult from the solver. None = not yet run.
    route_polylines : {(from_node, to_node): [(lat,lng), …]}.
                      Node 0 = depot. None = straight-line fallback.
    pending_click   : {lat, lng} of an unconfirmed map click.
    """

    # --- Collect all valid coordinate points to fit the view ---
    valid_lats, valid_lngs = [], []
    if destination:
        valid_lats.append(destination["lat"])
        valid_lngs.append(destination["lng"])
    for s in stops:
        if s.get("lat") is not None and s.get("lng") is not None:
            valid_lats.append(s["lat"])
            valid_lngs.append(s["lng"])
    if pending_click:
        valid_lats.append(pending_click["lat"])
        valid_lngs.append(pending_click["lng"])

    if valid_lats:
        center = [sum(valid_lats) / len(valid_lats), sum(valid_lngs) / len(valid_lngs)]
        zoom = 12
    else:
        center = [36.8987, 30.8005]  # Antalya Airport (AYT) default
        zoom = 11

    fmap = folium.Map(location=center, zoom_start=zoom, tiles="OpenStreetMap")

    if len(valid_lats) > 1:
        margin = 0.005
        fmap.fit_bounds([
            [min(valid_lats) - margin, min(valid_lngs) - margin],
            [max(valid_lats) + margin, max(valid_lngs) + margin],
        ])

    # --- Build node→vehicle assignment map from result ---
    node_to_vehicle: dict[int, int] = {}
    if result:
        for route in result.routes:
            for node in route.stop_indices:
                if node != 0:
                    node_to_vehicle[node] = route.vehicle_id

    # --- Hub marker ---
    if destination:
        folium.Marker(
            location=[destination["lat"], destination["lng"]],
            popup=folium.Popup(
                f"<b>🏢 {destination['name']}</b><br><i>Hub / Destination</i>",
                max_width=200,
            ),
            tooltip=destination["name"],
            icon=folium.Icon(color="black", icon="home", prefix="fa"),
        ).add_to(fmap)

    # --- Street-following route edges (post-optimisation) ---
    if result and destination:
        # Mirrors the node ordering used by the optimizer: 0=depot, i+1=stops[i]
        valid_stops = [
            s for s in stops
            if s.get("lat") is not None and s.get("lng") is not None
        ]
        all_locations: list[tuple[float, float]] = [
            (destination["lat"], destination["lng"])
        ] + [(s["lat"], s["lng"]) for s in valid_stops]

        def _node_latlon(n: int) -> tuple[float, float]:
            if 0 <= n < len(all_locations):
                return all_locations[n]
            return (destination["lat"], destination["lng"])

        for route in result.routes:
            color = _route_color(route.vehicle_id)
            nodes = route.stop_indices

            for i in range(len(nodes) - 1):
                from_node, to_node = nodes[i], nodes[i + 1]
                key = (from_node, to_node)

                if route_polylines and key in route_polylines:
                    coords = route_polylines[key]
                else:
                    coords = [_node_latlon(from_node), _node_latlon(to_node)]

                folium.PolyLine(
                    locations=coords,
                    color=color,
                    weight=5,
                    opacity=0.85,
                    tooltip=(
                        f"Shuttle {route.vehicle_id + 1} — "
                        f"{route.total_passengers} pax ({route.occupancy_pct}% full)"
                    ),
                ).add_to(fmap)

    # --- Stop markers ---
    for i, s in enumerate(stops):
        if s.get("lat") is None or s.get("lng") is None:
            continue

        node_idx = i + 1
        vid = node_to_vehicle.get(node_idx)
        color = _route_color(vid) if vid is not None else _UNASSIGNED_COLOR
        shuttle_label = f"Shuttle {vid + 1}" if vid is not None else "Not yet optimised"

        popup_html = (
            f"<b>{s['name']}</b><br>"
            f"Passengers: {s['passengers']}<br>"
            f"{shuttle_label}"
        )
        folium.CircleMarker(
            location=[s["lat"], s["lng"]],
            radius=10,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"{s['name']} ({s['passengers']} pax) · {shuttle_label}",
        ).add_to(fmap)

    # --- Pending click marker ---
    if pending_click:
        folium.Marker(
            location=[pending_click["lat"], pending_click["lng"]],
            tooltip="New node — fill in the form below ↓",
            icon=folium.Icon(color="orange", icon="plus", prefix="fa"),
        ).add_to(fmap)

    # --- Legend (post-optimisation only) ---
    if result and result.routes:
        legend_items = "".join(
            f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">'
            f'<div style="width:14px;height:14px;border-radius:50%;'
            f'background:{_route_color(r.vehicle_id)};"></div>'
            f'<span>Shuttle {r.vehicle_id + 1} — {r.total_passengers} pax '
            f'({r.occupancy_pct}%)</span></div>'
            for r in result.routes
        )
        legend_html = f"""
        <div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
                    padding:12px 16px;border-radius:8px;
                    box-shadow:0 2px 8px rgba(0,0,0,0.25);
                    font-family:sans-serif;font-size:13px;max-width:240px;">
          <b style="font-size:14px;">Route Legend</b>
          <div style="margin-top:8px;">{legend_items}</div>
        </div>
        """
        fmap.get_root().html.add_child(folium.Element(legend_html))

    return fmap

