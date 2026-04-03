"""
maps_client.py
Google Maps API wrapper: geocoding and distance matrix (batched).
"""

import math
import googlemaps


class MapsClient:
    """Thin wrapper around the googlemaps Python client."""

    # Distance Matrix API allows up to 10 origins × 10 destinations per request.
    _BATCH_SIZE = 10

    def __init__(self, api_key: str):
        if not api_key or not api_key.strip():
            raise ValueError("A valid Google Maps API key is required.")
        self._client = googlemaps.Client(key=api_key.strip())

    # ------------------------------------------------------------------
    # Geocoding
    # ------------------------------------------------------------------

    def geocode(self, address: str) -> tuple[float, float]:
        """Return (latitude, longitude) for a given address string."""
        results = self._client.geocode(address)
        if not results:
            raise ValueError(f"Could not geocode address: '{address}'")
        loc = results[0]["geometry"]["location"]
        return float(loc["lat"]), float(loc["lng"])

    # ------------------------------------------------------------------
    # Distance matrix
    # ------------------------------------------------------------------

    def build_distance_matrix(
        self,
        locations: list[tuple[float, float]],
        mode: str = "driving",
    ) -> list[list[int]]:
        """
        Build an N×N driving-distance matrix (values in metres as integers).

        The Distance Matrix API caps at 10 origins × 10 destinations per
        request, so large sets are requested in batches and assembled here.

        Parameters
        ----------
        locations : list of (lat, lng) tuples; index 0 is the depot/destination.
        mode      : travel mode passed to the API ("driving", "walking", …).

        Returns
        -------
        N×N list[list[int]] where matrix[i][j] is metres from location i to j.
        """
        n = len(locations)
        matrix: list[list[int]] = [[0] * n for _ in range(n)]

        lat_lng_strs = [f"{lat},{lng}" for lat, lng in locations]

        num_batches = math.ceil(n / self._BATCH_SIZE)

        for row_batch in range(num_batches):
            origins_slice = lat_lng_strs[
                row_batch * self._BATCH_SIZE: (row_batch + 1) * self._BATCH_SIZE
            ]
            origins_start = row_batch * self._BATCH_SIZE

            for col_batch in range(num_batches):
                dests_slice = lat_lng_strs[
                    col_batch * self._BATCH_SIZE: (col_batch + 1) * self._BATCH_SIZE
                ]
                dests_start = col_batch * self._BATCH_SIZE

                response = self._client.distance_matrix(
                    origins=origins_slice,
                    destinations=dests_slice,
                    mode=mode,
                )

                for r_idx, row in enumerate(response["rows"]):
                    for c_idx, element in enumerate(row["elements"]):
                        abs_row = origins_start + r_idx
                        abs_col = dests_start + c_idx

                        if element["status"] == "OK":
                            matrix[abs_row][abs_col] = element["distance"]["value"]
                        else:
                            # Fall back to a large penalty distance so the solver
                            # can still produce a feasible solution.
                            matrix[abs_row][abs_col] = 999_999_999

        return matrix
