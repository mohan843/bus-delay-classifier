# Bus Delay Classifier

A public-transportation machine-learning mini-project that classifies a bus trip as on-time, minor-delay, or major-delay using route length, traffic, rain, peak-hour status, and previous delay.

## Overview

Arrival-delay classification can help passengers plan and operators prioritize interventions. This project demonstrates a transparent k-nearest-neighbor workflow with feature standardization.

## Problem statement

Given trip and context features, predict a broad arrival-delay category for an upcoming bus trip.

## Features

- K-nearest-neighbor classification from scratch.
- Standardized features so units do not dominate distance.
- Configurable training data and CLI inference.
- CSV sample data, tests, and accuracy reporting.
- Standard-library-only implementation.

## Technology stack

- Python 3.10+
- K-nearest neighbors and Euclidean distance
- Feature standardization
- CSV, `argparse`, and `pathlib`
- `unittest` for automated testing

## Installation

```bash
git clone https://github.com/mohan843/bus-delay-classifier.git
cd bus-delay-classifier
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

No third-party runtime dependencies are required.

## Execution

```bash
python classifier.py --route-km 14 --traffic-index 70 --rain-mm 4 --scheduled-peak 1 --previous-delay-min 12
```

## Testing

```bash
python -m unittest discover -s tests -v
```

## Sample output

```text
Training accuracy: 87.5%
Predicted delay class: major
```

## Limitations

The dataset is synthetic and small. Real transit models need route geometry, stop dwell time, vehicle telemetry, incidents, time-aware validation, and uncertainty estimates. Predictions should support operations rather than promise exact arrival times.

## Future improvements

- Add stop-level and route-level temporal features.
- Compare tree ensembles and time-series models.
- Predict delay minutes with probabilistic intervals.
- Integrate live traffic and weather feeds.
- Build passenger-facing alerts and operator dashboards.
