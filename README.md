# Kalman Filter Tracking

Tracking experiments with linear Kalman filtering, fixed-lag smoothing, and an extended Kalman filter for a nonlinear unicycle model.

## Highlights

- Implements a constant-velocity linear Kalman filter.
- Adds fixed-lag smoothing from the forward filter pass.
- Simulates and tracks nonlinear unicycle motion with an EKF.
- Exports XY trajectory plots and component-level diagnostics.

## Repository Layout

- `tracking_filters.py` - filtering, smoothing, simulation, and plotting code.
- `data/observations.npy` - observation sequence.
- `examples/` - generated trajectory and component plots.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python tracking_filters.py
```

