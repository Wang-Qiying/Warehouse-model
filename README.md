# Refrigerated Warehouse Thermal Model Utilities

This repository contains a compact Python module for constructing the thermal
parameters and occupancy schedule of a small refrigerated warehouse model.

The code is extracted from a refrigerated-warehouse indoor temperature-control
case study. It keeps only the reusable modeling layer:

- geometry and envelope heat-transfer parameters;
- equivalent thermal capacity calculation;
- BEAR-compatible parameter dictionary construction;
- pharmaceutical cold-storage occupancy and metabolic heat schedule.

It does **not** include control algorithms, weather datasets, raw simulation
results, private paths, or experiment-specific notebooks.

## Files

```text
warehouse_thermal_model.py   # Main reusable module
requirements.txt             # Minimal Python dependencies
README.md                    # Usage and repository description
```

## Installation

```bash
pip install -r requirements.txt
```

## Basic usage

```python
import pandas as pd
from warehouse_thermal_model import (
    WarehouseThermalConfig,
    build_bear_warehouse_parameter,
)

weather = pd.read_csv("WD_true_data.csv")

config = WarehouseThermalConfig(
    floor_area_m2=100.0,
    height_m=4.0,
    max_power_w=8000.0,
)

parameter, derived = build_bear_warehouse_parameter(
    weather,
    config=config,
    people_per_zone=[0, 0, 2],
    steps_per_hour=4,  # 15-min interval
)

print(derived["room_heat_capacity_j_k"])
print(parameter.keys())
```

If the BEAR package is installed, the returned dictionary can be passed to
`BuildingEnvReal`:

```python
from BEAR.Env.env_building import BuildingEnvReal

env = BuildingEnvReal(parameter)
```

## Weather-data format

The input weather table must contain:

| Column | Unit | Description |
|---|---:|---|
| `temp_air` | degC | Outdoor air temperature |
| `ghi` | W/m2 | Global horizontal irradiance |

Optional:

| Column | Unit | Description |
|---|---:|---|
| `ground_temp` | degC | Ground or surface temperature; if absent, `temp_air` is used |

## License

Add a license file before public release. For academic code intended for broad
reuse, the MIT License is often a simple permissive choice. If you need stronger
requirements for derivative works to remain open source, consider GPL-family
licenses instead.
