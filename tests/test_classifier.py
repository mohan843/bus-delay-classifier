import unittest
from pathlib import Path

from classifier import KNNClassifier, StandardScaler, accuracy, load_csv, train

DATA = Path(__file__).parents[1] / "data" / "bus_trips.csv"


class BusDelayTests(unittest.TestCase):
    def test_dataset_loads(self):
        rows, labels = load_csv(DATA)
        self.assertEqual(len(rows), 16)
        self.assertEqual(set(labels), {"on-time", "minor", "major"})

    def test_scaler_centers_features(self):
        transformed = StandardScaler().fit([[1, 10], [3, 20]]).transform([[1, 10], [3, 20]])
        self.assertAlmostEqual(sum(row[0] for row in transformed), 0.0)

    def test_classifier_predicts_delay_profiles(self):
        model, scaler, score = train(DATA)
        self.assertGreaterEqual(score, 0.80)
        self.assertEqual(model.predict_one(scaler.transform([[4, 20, 0, 0, 0]])[0]), "on-time")
        self.assertEqual(model.predict_one(scaler.transform([[14, 70, 4, 1, 12]])[0]), "major")

    def test_invalid_input_rejected(self):
        with self.assertRaises(ValueError):
            KNNClassifier().fit([], [])

    def test_accuracy(self):
        self.assertEqual(accuracy(["on-time", "major"], ["on-time", "on-time"]), 0.5)


if __name__ == "__main__":
    unittest.main()
