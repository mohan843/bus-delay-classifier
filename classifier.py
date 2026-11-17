"""Classify bus arrival delay with a small k-nearest-neighbor model."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

FEATURES = ("route_km", "traffic_index", "rain_mm", "scheduled_peak", "previous_delay_min")


class StandardScaler:
    def fit(self, rows: Sequence[Sequence[float]]) -> "StandardScaler":
        if not rows:
            raise ValueError("At least one row is required")
        self.means = [sum(row[i] for row in rows) / len(rows) for i in range(len(rows[0]))]
        self.scales = []
        for i, mean in enumerate(self.means):
            variance = sum((row[i] - mean) ** 2 for row in rows) / len(rows)
            self.scales.append(variance ** 0.5 or 1.0)
        return self

    def transform(self, rows: Sequence[Sequence[float]]) -> list[list[float]]:
        return [[(value - mean) / scale for value, mean, scale in zip(row, self.means, self.scales)] for row in rows]


class KNNClassifier:
    def __init__(self, k: int = 3) -> None:
        if k < 1:
            raise ValueError("k must be positive")
        self.k = k

    def fit(self, rows: Sequence[Sequence[float]], labels: Sequence[str]) -> "KNNClassifier":
        if not rows or len(rows) != len(labels):
            raise ValueError("Rows and labels must be non-empty and equally sized")
        self.rows, self.labels = list(rows), list(labels)
        return self

    def predict_one(self, row: Sequence[float]) -> str:
        distances = sorted((sum((a - b) ** 2 for a, b in zip(row, train_row)) ** 0.5, label) for train_row, label in zip(self.rows, self.labels))
        votes = {}
        for _, label in distances[: self.k]:
            votes[label] = votes.get(label, 0) + 1
        return max(votes, key=lambda label: (votes[label], label))

    def predict(self, rows: Sequence[Sequence[float]]) -> list[str]:
        return [self.predict_one(row) for row in rows]


def load_csv(path: str | Path) -> tuple[list[list[float]], list[str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [[float(row[name]) for name in FEATURES] for row in rows], [row["delay_class"] for row in rows]


def accuracy(actual: Sequence[str], predicted: Sequence[str]) -> float:
    return sum(a == p for a, p in zip(actual, predicted)) / len(actual)


def train(path: str | Path) -> tuple[KNNClassifier, StandardScaler, float]:
    rows, labels = load_csv(path)
    scaler = StandardScaler().fit(rows)
    transformed = scaler.transform(rows)
    model = KNNClassifier().fit(transformed, labels)
    return model, scaler, accuracy(labels, model.predict(transformed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify expected bus arrival delay.")
    parser.add_argument("--data", default="data/bus_trips.csv")
    parser.add_argument("--route-km", type=float, required=True)
    parser.add_argument("--traffic-index", type=float, required=True)
    parser.add_argument("--rain-mm", type=float, required=True)
    parser.add_argument("--scheduled-peak", type=float, required=True)
    parser.add_argument("--previous-delay-min", type=float, required=True)
    args = parser.parse_args()
    model, scaler, score = train(args.data)
    row = [[args.route_km, args.traffic_index, args.rain_mm, args.scheduled_peak, args.previous_delay_min]]
    print(f"Training accuracy: {score:.1%}")
    print(f"Predicted delay class: {model.predict_one(scaler.transform(row)[0])}")
