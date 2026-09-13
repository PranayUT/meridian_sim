"""Load GPS routes into the Gazebo world's local ENU frame."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

WORLD_LATITUDE_DEG = 30.326139
WORLD_LONGITUDE_DEG = -98.148264
WGS84_A = 6_378_137.0
WGS84_E2 = 6.69437999014e-3
MAX_ROUTE_BYTES = 16 * 1024 * 1024


class RouteFileError(ValueError):
    """A route file does not contain a usable WGS84 path."""


def wgs84_to_world(longitude: float, latitude: float) -> tuple[float, float]:
    """Project a nearby WGS84 point into the simulator's ENU frame."""
    latitude0 = math.radians(WORLD_LATITUDE_DEG)
    longitude0 = math.radians(WORLD_LONGITUDE_DEG)
    sin_latitude = math.sin(latitude0)
    denominator = math.sqrt(1.0 - WGS84_E2 * sin_latitude * sin_latitude)
    prime_vertical_radius = WGS84_A / denominator
    meridian_radius = WGS84_A * (1.0 - WGS84_E2) / denominator**3
    east = (
        math.radians(longitude) - longitude0
    ) * prime_vertical_radius * math.cos(latitude0)
    north = (math.radians(latitude) - latitude0) * meridian_radius
    return east, north


def _coordinates_from_kml(data: bytes) -> list[tuple[float, float]]:
    try:
        root = ET.fromstring(data)
    except ET.ParseError as error:
        raise RouteFileError(f"KML is not valid XML: {error}") from error
    coordinate_nodes = [element for element in root.iter() if element.tag.endswith("coordinates")]
    if not coordinate_nodes:
        raise RouteFileError("KML has no path coordinates")
    # Use the longest coordinate sequence. Google Earth can include point
    # placemarks beside the path.
    sequences: list[list[tuple[float, float]]] = []
    for node in coordinate_nodes:
        points: list[tuple[float, float]] = []
        for value in (node.text or "").split():
            fields = value.split(",")
            if len(fields) < 2:
                continue
            try:
                points.append((float(fields[0]), float(fields[1])))
            except ValueError as error:
                raise RouteFileError("KML contains a non-numeric coordinate") from error
        sequences.append(points)
    return max(sequences, key=len)


def _coordinates_from_json(path: Path) -> list[tuple[float, float]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RouteFileError(f"cannot read GPS JSON: {error}") from error
    rows = raw.get("latlon") if isinstance(raw, dict) else None
    if rows is None and isinstance(raw, dict):
        rows = raw.get("wps")
    if not isinstance(rows, list):
        raise RouteFileError("GPS JSON needs a latlon or wps list")
    coordinates = []
    for row in rows:
        try:
            if isinstance(row, dict):
                latitude, longitude = float(row["lat"]), float(row["lon"])
            else:
                latitude, longitude = float(row[0]), float(row[1])
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise RouteFileError("GPS JSON contains an invalid waypoint") from error
        coordinates.append((longitude, latitude))
    return coordinates


def load_route(path: Path) -> list[tuple[float, float]]:
    """Load KMZ, KML, or Meridian GPS JSON and return world XY points."""
    path = path.expanduser().resolve()
    suffix = path.suffix.lower()
    if suffix == ".kmz":
        try:
            with zipfile.ZipFile(path) as archive:
                candidates = [item for item in archive.infolist() if item.filename.lower().endswith(".kml")]
                if not candidates:
                    raise RouteFileError("KMZ has no KML document")
                member = max(candidates, key=lambda item: item.file_size)
                if member.file_size > MAX_ROUTE_BYTES:
                    raise RouteFileError("KMZ route document is too large")
                coordinates = _coordinates_from_kml(archive.read(member))
        except (OSError, zipfile.BadZipFile) as error:
            raise RouteFileError(f"cannot read KMZ: {error}") from error
    elif suffix == ".kml":
        try:
            data = path.read_bytes()
        except OSError as error:
            raise RouteFileError(f"cannot read KML: {error}") from error
        if len(data) > MAX_ROUTE_BYTES:
            raise RouteFileError("KML route document is too large")
        coordinates = _coordinates_from_kml(data)
    elif suffix == ".json":
        coordinates = _coordinates_from_json(path)
    else:
        raise RouteFileError("route file must be KMZ, KML, or GPS JSON")
    if len(coordinates) < 2:
        raise RouteFileError("route needs at least two GPS points")
    result = []
    for longitude, latitude in coordinates:
        if not (-180.0 <= longitude <= 180.0 and -90.0 <= latitude <= 90.0):
            raise RouteFileError("route contains a coordinate outside WGS84 limits")
        point = wgs84_to_world(longitude, latitude)
        if not result or math.hypot(point[0] - result[-1][0], point[1] - result[-1][1]) > 0.01:
            result.append(point)
    if len(result) < 2:
        raise RouteFileError("route has fewer than two distinct GPS points")
    if any(abs(x) > 256.0 or abs(y) > 256.0 for x, y in result):
        raise RouteFileError("route leaves the 512 m simulator terrain")
    return result


def find_default_route(project_root: Path) -> Path | None:
    """Select the only supported route in paths, if there is one."""
    directory = project_root / "paths"
    if not directory.is_dir():
        return None
    candidates = sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in {".kmz", ".kml", ".json"}
    )
    return candidates[0] if len(candidates) == 1 else None

