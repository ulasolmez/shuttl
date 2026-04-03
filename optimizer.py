"""
optimizer.py
Capacitated Vehicle Routing Problem (CVRP) solver using Google OR-Tools.

Node 0 is always the depot (destination hub). Passenger stops are nodes 1…N.
The solver minimises:
  - number of vehicles used  (via a large fixed cost per vehicle)
  - total driving distance   (via arc cost = metres)
These two objectives together naturally maximise per-shuttle occupancy.
"""

from __future__ import annotations
from dataclasses import dataclass, field

from ortools.constraint_solver import pywrapcp, routing_enums_pb2


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RouteResult:
    vehicle_id: int
    stop_indices: list[int]          # indices into the original stops list (0 = depot)
    stop_names: list[str]
    total_passengers: int
    total_distance_m: int
    occupancy_pct: float             # total_passengers / vehicle_capacity * 100


@dataclass
class OptimizationResult:
    routes: list[RouteResult] = field(default_factory=list)
    vehicles_used: int = 0
    total_passengers: int = 0
    total_distance_m: int = 0
    avg_occupancy_pct: float = 0.0
    solver_status: str = ""


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class ShuttleOptimizer:
    """
    Solves the CVRP for hub-and-spoke shuttle routing.

    Parameters
    ----------
    distance_matrix : N×N matrix (metres); index 0 is the depot.
    demands         : list of N ints; demands[0] must be 0 (depot has no demand).
    stop_names      : list of N strings corresponding to each node.
    num_vehicles    : maximum number of shuttles available.
    vehicle_capacity: maximum passengers per shuttle.
    max_route_distance_m : maximum total route length per shuttle in metres.
                           0 means no limit.
    time_limit_sec  : seconds the solver is allowed to search.
    """

    # A fixed cost added per vehicle *used* — large enough to penalise opening
    # an extra shuttle, but not so large it overrides feasibility.
    _FIXED_VEHICLE_COST_MULTIPLIER = 2

    def __init__(
        self,
        distance_matrix: list[list[int]],
        demands: list[int],
        stop_names: list[str],
        num_vehicles: int,
        vehicle_capacity: int,
        max_route_distance_m: int = 0,
        time_limit_sec: int = 30,
    ):
        if len(distance_matrix) != len(demands):
            raise ValueError("distance_matrix size must equal len(demands).")
        if demands[0] != 0:
            raise ValueError("demands[0] (depot) must be 0.")

        total_passengers = sum(demands)
        max_capacity = num_vehicles * vehicle_capacity
        if total_passengers > max_capacity:
            raise ValueError(
                f"Total passengers ({total_passengers}) exceeds fleet capacity "
                f"({num_vehicles} vehicles × {vehicle_capacity} seats = {max_capacity}). "
                "Add more vehicles or increase capacity."
            )

        self._matrix = distance_matrix
        self._demands = demands
        self._stop_names = stop_names
        self._num_vehicles = num_vehicles
        self._vehicle_capacity = vehicle_capacity
        self._max_route_distance_m = max_route_distance_m
        self._time_limit_sec = time_limit_sec
        self._n = len(distance_matrix)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self) -> OptimizationResult:
        manager = pywrapcp.RoutingIndexManager(
            self._n,
            self._num_vehicles,
            0,  # depot index
        )
        routing = pywrapcp.RoutingModel(manager)

        # --- Arc cost (distance) callback ---
        # Depot→stop arcs are free (one-way pickup routing: stops→hub only).
        def distance_callback(from_index: int, to_index: int) -> int:
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            if from_node == 0:
                return 0
            return self._matrix[from_node][to_node]

        transit_cb_idx = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_idx)

        # --- Demand (passenger) callback ---
        def demand_callback(from_index: int) -> int:
            node = manager.IndexToNode(from_index)
            return self._demands[node]

        demand_cb_idx = routing.RegisterUnaryTransitCallback(demand_callback)

        # --- Capacity dimension ---
        routing.AddDimensionWithVehicleCapacity(
            demand_cb_idx,
            0,                                           # no slack
            [self._vehicle_capacity] * self._num_vehicles,
            True,                                        # fix_start_cumul_to_zero
            "Capacity",
        )

        # --- Fixed cost per vehicle used — drives solver to fill shuttles ---
        # Exclude penalty values (999_999_999) from max_arc so the fixed cost
        # is scaled to real distances only.
        _PENALTY = 999_999_999
        real_arcs = [v for row in self._matrix for v in row if v < _PENALTY]
        max_arc = max(real_arcs) if real_arcs else 1
        fixed_cost = max_arc * self._FIXED_VEHICLE_COST_MULTIPLIER
        routing.SetFixedCostOfAllVehicles(fixed_cost)

        # --- Max route distance per vehicle ---
        if self._max_route_distance_m > 0:
            # Distance dimension measures real inbound driving distance.
            # Depot→stop arcs must be zeroed here too — the raw matrix contains
            # 999_999_999 for hub→stop when the hub is an airport / restricted
            # exit. Without the zero, the first arc blows past any km cap.
            def real_distance_callback(from_index: int, to_index: int) -> int:
                from_node = manager.IndexToNode(from_index)
                if from_node == 0:
                    return 0
                return self._matrix[from_node][manager.IndexToNode(to_index)]

            real_transit_idx = routing.RegisterTransitCallback(real_distance_callback)
            routing.AddDimensionWithVehicleCapacity(
                real_transit_idx,
                0,                                              # no slack
                [self._max_route_distance_m] * self._num_vehicles,
                True,                                           # fix cumul to zero
                "Distance",
            )

        # --- Search parameters ---
        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_params.time_limit.FromSeconds(self._time_limit_sec)

        solution = routing.SolveWithParameters(search_params)

        if solution is None:
            return OptimizationResult(solver_status="NO_SOLUTION")

        return self._extract_solution(manager, routing, solution)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_solution(
        self,
        manager: pywrapcp.RoutingIndexManager,
        routing: pywrapcp.RoutingModel,
        solution,
    ) -> OptimizationResult:
        routes: list[RouteResult] = []
        total_dist = 0

        for vehicle_id in range(self._num_vehicles):
            index = routing.Start(vehicle_id)

            # Skip vehicles that are not used (only at depot from start to end).
            if routing.IsEnd(solution.Value(routing.NextVar(index))):
                continue

            stop_indices: list[int] = []
            stop_names: list[str] = []
            route_passengers = 0
            # Skip the leading depot node — route is displayed as stops → hub.
            first = True

            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                if not (first and node == 0):
                    stop_indices.append(node)
                    stop_names.append(self._stop_names[node])
                    route_passengers += self._demands[node]
                first = False
                index = solution.Value(routing.NextVar(index))

            # Append hub at end to close the inbound route.
            stop_indices.append(manager.IndexToNode(index))
            stop_names.append(self._stop_names[manager.IndexToNode(index)])

            # Distance: sum real arcs only (stop→stop and last_stop→hub).
            # Re-walk the OR-Tools solution to pick up inbound arc costs.
            route_dist = 0
            idx = routing.Start(vehicle_id)
            idx = solution.Value(routing.NextVar(idx))  # skip depot→first_stop
            while not routing.IsEnd(idx):
                nxt = solution.Value(routing.NextVar(idx))
                route_dist += self._matrix[
                    manager.IndexToNode(idx)][manager.IndexToNode(nxt)]
                idx = nxt

            total_dist += route_dist
            occ_pct = round(route_passengers / self._vehicle_capacity * 100, 1)

            routes.append(RouteResult(
                vehicle_id=vehicle_id,
                stop_indices=stop_indices,
                stop_names=stop_names,
                total_passengers=route_passengers,
                total_distance_m=route_dist,
                occupancy_pct=occ_pct,
            ))

        total_passengers = sum(r.total_passengers for r in routes)
        avg_occ = (
            round(sum(r.occupancy_pct for r in routes) / len(routes), 1)
            if routes else 0.0
        )

        return OptimizationResult(
            routes=routes,
            vehicles_used=len(routes),
            total_passengers=total_passengers,
            total_distance_m=total_dist,
            avg_occupancy_pct=avg_occ,
            solver_status="SOLUTION_FOUND",
        )
