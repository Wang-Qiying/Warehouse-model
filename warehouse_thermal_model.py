"""
Open-source refrigerated warehouse thermal-parameter and occupancy utilities.

This module extracts the warehouse thermal-parameter construction and the
pharmaceutical cold-storage occupancy schedule from the user's simulation
workflow. It is intentionally independent of the control algorithms. The
returned parameter dictionary is compatible with BEAR's ``BuildingEnvReal``
interface, while the module itself only requires NumPy and pandas.

Typical use
-----------
>>> import pandas as pd
>>> from warehouse_thermal_model import (
...     WarehouseThermalConfig,
...     build_bear_warehouse_parameter,
... )
>>> weather = pd.read_csv("WD_true_data.csv")
>>> cfg = WarehouseThermalConfig()
>>> parameter, derived = build_bear_warehouse_parameter(
...     weather,
...     config=cfg,
...     people_per_zone=[0, 0, 2],
...     steps_per_hour=4,
... )
>>> # Optional BEAR-side usage:
>>> # from BEAR.Env.env_building import BuildingEnvReal
>>> # env = BuildingEnvReal(parameter)

Notes
-----
Required weather columns:
    - ``temp_air``: outdoor air temperature in degree Celsius.
    - ``ghi``: global horizontal irradiance in W/m^2.

Optional weather column:
    - ``ground_temp``: ground or surface temperature in degree Celsius.
      If absent, ``temp_air`` is used as a fallback.

The default warehouse represents a small 100 m^2 cold-storage room with a
2--6 degree Celsius operating range and an 8 kW temperature-control system.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Mapping, Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WarehouseThermalConfig:
    """Physical and BEAR-interface parameters for the refrigerated warehouse."""

    # Geometry
    floor_area_m2: float = 100.0
    height_m: float = 4.0
    outdoor_wall_area_m2: float = 160.0
    window_area_m2: float = 0.0

    # Envelope heat-transfer coefficients, W/(m^2 K)
    u_outwall_w_m2k: float = 0.35
    u_roof_w_m2k: float = 0.30
    u_floor_w_m2k: float = 0.47
    u_window_w_m2k: float = 0.36

    # Ground and solar terms
    ground_weight: float = 0.4
    shgc: float = 0.252
    shgc_weight: float = 0.01

    # Air and equivalent thermal mass
    air_specific_heat_j_kgk: float = 1005.0
    air_density_kg_m3: float = 1.27
    thermal_mass_multiplier: float = 6.0

    # HVAC and simulation settings
    max_power_w: float = 8000.0
    target_temperature_c: float = 4.0
    temp_range_c: tuple[float, float] = (-10.0, 15.0)
    ac_map: float = 1.0

    # BEAR reward weights, kept for interface compatibility
    gamma: tuple[float, float] = (0.001, 0.999)

    @property
    def roof_area_m2(self) -> float:
        """Roof area, assumed equal to floor area for the base cuboid model."""
        return self.floor_area_m2

    @property
    def air_heat_capacity_j_k(self) -> float:
        """Heat capacity of indoor air only."""
        return (
            self.air_specific_heat_j_kgk
            * self.air_density_kg_m3
            * self.floor_area_m2
            * self.height_m
        )

    @property
    def room_heat_capacity_j_k(self) -> float:
        """Equivalent room heat capacity including empirical thermal inertia."""
        return self.thermal_mass_multiplier * self.air_heat_capacity_j_k

    @property
    def ua_outdoor_w_k(self) -> float:
        """Equivalent heat conductance from outdoor air to the room."""
        return (
            self.outdoor_wall_area_m2 * self.u_outwall_w_m2k
            + self.roof_area_m2 * self.u_roof_w_m2k
            + self.window_area_m2 * self.u_window_w_m2k
        )

    @property
    def ua_ground_w_k(self) -> float:
        """Weighted equivalent heat conductance from ground/surface temperature."""
        return self.floor_area_m2 * self.u_floor_w_m2k * self.ground_weight


def make_pharma_cold_occupancy(
    n_hours: int = 8760,
    people_per_zone: Iterable[float] | None = None,
    meta_active_w_per_person: float = 150.0,
    meta_patrol_w_per_person: float = 100.0,
    steps_per_hour: int = 1,
    start_day_of_week: int = 0,
    hold_within_hour: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Create occupancy parameters for a pharmaceutical cold-storage warehouse.

    Parameters
    ----------
    n_hours:
        Number of simulated hours.
    people_per_zone:
        Peak population in each zone. The default is ``[1, 1, 2]`` for
        office, refined-storage, and bulk-storage zones.
    meta_active_w_per_person:
        Metabolic heat rate during active working hours, W/person.
    meta_patrol_w_per_person:
        Metabolic heat rate during night patrol, W/person.
    steps_per_hour:
        Number of control/simulation steps per hour. For a 15-min interval,
        use ``steps_per_hour=4``.
    start_day_of_week:
        Day index of the first simulated day. ``0`` means Monday and ``6``
        means Sunday.
    hold_within_hour:
        If ``True``, the hourly activity value is assigned to all sub-hourly
        steps within that hour. If ``False``, only the first sub-hourly step
        of each hour is assigned, which reproduces the pulse-style behavior
        of some legacy BEAR scripts.

    Returns
    -------
    full_occ:
        Array of peak population values per zone.
    activity_schedule:
        Step-wise metabolic heat schedule, W/person. A zero value means that
        the warehouse is unoccupied at that period.

    Schedule
    --------
    Weekdays:
        00:00--05:59: night patrol, ``meta_patrol_w_per_person``;
        06:00--07:59: empty;
        08:00--15:59: active shift, ``meta_active_w_per_person``;
        16:00--21:59: empty;
        22:00--23:59: night patrol, ``meta_patrol_w_per_person``.

    Weekends:
        08:00--15:59: duty shift with half active metabolic rate;
        otherwise empty.
    """
    if n_hours <= 0:
        raise ValueError("n_hours must be positive.")
    if steps_per_hour <= 0:
        raise ValueError("steps_per_hour must be positive.")

    if people_per_zone is None:
        people_per_zone = [1.0, 1.0, 2.0]

    full_occ = np.asarray(list(people_per_zone), dtype=float)
    n_steps = int(n_hours * steps_per_hour)
    steps_per_day = int(24 * steps_per_hour)
    schedule = np.zeros(n_steps, dtype=float)

    for step in range(n_steps):
        day_of_week = (start_day_of_week + step // steps_per_day) % 7
        hour_of_day = (step % steps_per_day) // steps_per_hour
        subhour_index = step % steps_per_hour
        is_weekend = day_of_week >= 5

        if not hold_within_hour and subhour_index != 0:
            continue

        value = 0.0
        if is_weekend:
            if 8 <= hour_of_day < 16:
                value = 0.5 * meta_active_w_per_person
        else:
            if 8 <= hour_of_day < 16:
                value = meta_active_w_per_person
            elif hour_of_day < 6 or hour_of_day >= 22:
                value = meta_patrol_w_per_person

        schedule[step] = value

    return full_occ, schedule


def _as_1d_float_array(values: Any, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def build_bear_warehouse_parameter(
    weather: pd.DataFrame | Mapping[str, Iterable[float]],
    config: WarehouseThermalConfig | None = None,
    people_per_zone: Iterable[float] | None = None,
    steps_per_hour: int = 4,
    time_resolution_s: int | None = None,
    normalize_ghi_by_max: bool = True,
    hold_occupancy_within_hour: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the BEAR parameter dictionary for the refrigerated warehouse.

    Parameters
    ----------
    weather:
        DataFrame or mapping containing at least ``temp_air`` and ``ghi``.
        ``ground_temp`` is optional.
    config:
        Warehouse thermal configuration. If omitted, the default small cold
        warehouse configuration is used.
    people_per_zone:
        Peak population values per zone. For a single-room aggregate model,
        their sum is used as the effective number of occupants.
    steps_per_hour:
        Number of control/simulation steps per hour.
    time_resolution_s:
        Simulation step length in seconds. If omitted, it is set to
        ``3600 // steps_per_hour``.
    normalize_ghi_by_max:
        If ``True``, the GHI sequence passed to BEAR is normalized by its
        maximum value, following the original BEAR-compatible script.
    hold_occupancy_within_hour:
        Passed to :func:`make_pharma_cold_occupancy`.

    Returns
    -------
    parameter:
        Dictionary that can be passed to ``BEAR.Env.env_building.BuildingEnvReal``.
    derived:
        Dictionary of derived physical quantities, useful for reporting and
        reproducibility.
    """
    if config is None:
        config = WarehouseThermalConfig()
    if steps_per_hour <= 0:
        raise ValueError("steps_per_hour must be positive.")
    if time_resolution_s is None:
        time_resolution_s = int(3600 // steps_per_hour)

    weather_df = pd.DataFrame(weather)
    for col in ("temp_air", "ghi"):
        if col not in weather_df.columns:
            raise KeyError(f"weather must contain a '{col}' column.")

    out_temp = _as_1d_float_array(weather_df["temp_air"].to_numpy(), "temp_air")
    ghi_raw = _as_1d_float_array(weather_df["ghi"].to_numpy(), "ghi")
    if "ground_temp" in weather_df.columns:
        ground_temp = _as_1d_float_array(weather_df["ground_temp"].to_numpy(), "ground_temp")
    else:
        ground_temp = out_temp.copy()

    n_steps = len(out_temp)
    if len(ghi_raw) != n_steps or len(ground_temp) != n_steps:
        raise ValueError("weather columns must have the same length.")

    ghi_max = float(np.nanmax(ghi_raw))
    if normalize_ghi_by_max:
        if ghi_max <= 0:
            ghi_norm = np.zeros_like(ghi_raw, dtype=float)
        else:
            ghi_norm = ghi_raw / ghi_max
    else:
        ghi_norm = ghi_raw.copy()

    n_hours_needed = int(np.ceil(n_steps / steps_per_hour))
    full_occ, activity_schedule = make_pharma_cold_occupancy(
        n_hours=n_hours_needed,
        people_per_zone=people_per_zone,
        steps_per_hour=steps_per_hour,
        hold_within_hour=hold_occupancy_within_hour,
    )
    activity_schedule = activity_schedule[:n_steps].astype(float)

    total_people = float(np.sum(full_occ))
    c_room = config.room_heat_capacity_j_k
    ua_outdoor = config.ua_outdoor_w_k
    ua_ground = config.ua_ground_w_k

    # The original BEAR-compatible script scales SHGC by max GHI because the
    # GHI input to the environment is normalized.
    shgc_scaled = config.shgc * config.shgc_weight * (ghi_max if normalize_ghi_by_max else 1.0)

    roomnum = 1
    connectmap = np.array([[0.0, 1.0]], dtype=float)
    rctable = np.array([[0.0, ua_outdoor / c_room]], dtype=float)

    people_full = np.array([[total_people]], dtype=float)
    acweight = np.array([[config.ac_map * config.max_power_w]], dtype=float)
    window_term = np.array([[config.window_area_m2 * shgc_scaled]], dtype=float)

    weightcmap = np.concatenate(
        [
            people_full,
            np.array([[ua_ground]], dtype=float),
            np.zeros((1, 1), dtype=float),
            acweight,
            window_term,
        ],
        axis=-1,
    ) / c_room

    nonlinear = people_full / c_room

    parameter: dict[str, Any] = {
        "OutTemp": out_temp,
        "connectmap": connectmap,
        "RCtable": rctable,
        "roomnum": roomnum,
        "weightcmap": weightcmap,
        "target": np.array([config.target_temperature_c], dtype=float),
        "gamma": config.gamma,
        "time_resolution": int(time_resolution_s),
        "ghi": ghi_norm,
        "GroundTemp": ground_temp,
        "Occupancy": activity_schedule,
        "ACmap": np.array([config.ac_map], dtype=float),
        "max_power": float(config.max_power_w),
        "nonlinear": nonlinear,
        "temp_range": tuple(config.temp_range_c),
        "spacetype": "continuous",
    }

    derived: dict[str, Any] = {
        "config": asdict(config),
        "n_steps": n_steps,
        "steps_per_hour": int(steps_per_hour),
        "time_resolution_s": int(time_resolution_s),
        "total_people": total_people,
        "ghi_max_w_m2": ghi_max,
        "air_heat_capacity_j_k": config.air_heat_capacity_j_k,
        "room_heat_capacity_j_k": c_room,
        "ua_outdoor_w_k": ua_outdoor,
        "ua_ground_w_k": ua_ground,
        "shgc_scaled": shgc_scaled,
        "full_occ": full_occ,
    }

    return parameter, derived


__all__ = [
    "WarehouseThermalConfig",
    "make_pharma_cold_occupancy",
    "build_bear_warehouse_parameter",
]
