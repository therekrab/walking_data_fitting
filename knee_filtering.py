import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas
from scipy.signal import find_peaks
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

BASE_DIR = "forward_walking_molinaro"


def solve_aa_ellipse(x, y):
    design = np.column_stack([x**2, x, y, np.ones(x.shape)])
    sol, _, _, _ = np.linalg.lstsq(design, y**2)

    x_center = -sol[1] / (2 * sol[0])
    y_center = 0.5 * sol[2]

    b_squared = sol[3] + y_center**2 - (x_center**2) * sol[0]
    a_squared = -b_squared / sol[0]

    a = np.sqrt(a_squared)
    b = np.sqrt(b_squared)

    return (x_center, y_center, a, b)


def phase_variable(x, y):
    x_center, y_center, width, height = solve_aa_ellipse(x, y)
    return (
        0.5 * np.atan2((y - y_center) / height, (x - x_center) / width) / np.pi
    ) + 0.5


class Dataset:
    def __init__(self, speed, file_base):
        self.name = file_base
        self.angle_df = pandas.read_csv(f"{BASE_DIR}/{file_base}_off_angle_filt.csv")
        self.moment_df = pandas.read_csv(f"{BASE_DIR}/{file_base}_off_moment_filt.csv")

        self.time = self.angle_df.get("time").to_numpy()
        self.time = self.time - self.time[0]  # start at 0
        self.dt = np.diff(self.time)
        self.speed = speed

        # processing time!
        print(f"{file_base}")

        angle, _, _ = self.load("knee_angle_l")

        norm = angle - np.mean(angle)
        acf = np.correlate(norm, norm, mode="full")[len(norm) - 1 :]
        peaks, _ = find_peaks(acf)
        period_ticks = max(peaks, key=lambda p: acf[p])
        self.period = period_ticks * np.mean(self.dt)
        print(f"  Period: {period_ticks} ({self.period:.02}s)")

        cycles = np.resize(angle, (len(angle) // period_ticks, period_ticks))
        mean_cycle = np.mean(cycles, axis=0)

        peaks, _ = find_peaks(-mean_cycle)
        self.start = min(
            peaks, key=lambda p: mean_cycle[p]
        )  # should be first real cycle!
        print(f"  Start time: {self.start} ({self.start * 0.005:.03}s)")
        offset = self.start * np.mean(self.dt)
        shifted_time = self.time - offset
        self.phase = np.mod(shifted_time, self.period) / self.period

    def get_name(self):
        return f"{self.speed} meters per second"

    def load(self, name):
        x = self.angle_df.get(name).to_numpy()
        v = np.gradient(x, self.time)
        moment = self.moment_df.get(name + "_moment").to_numpy()
        if np.isnan(moment).any():
            print(f"[ERROR]: {self.name}.{name} has NaN moment")
            sys.exit(1)
        return (x, v, moment)

    def make_training_data(self):
        angle_left, rate_left, moment_left = self.load("knee_angle_l")
        angle_right, rate_right, _ = self.load("knee_angle_r")

        return (
            np.column_stack(
                [
                    angle_left,
                    rate_left,
                    angle_right,
                    rate_right,
                    np.full_like(self.time, self.speed),
                    np.full_like(self.time, 1 / self.speed),
                ]
            ),
            moment_left.reshape(-1, 1),  # one column
        )

    def graph(self, ax, model):
        inputs, correct = self.make_training_data()
        # knee_l, v_l, _ = self.load("knee_angle_l")
        # knee_r, v_r, _ = self.load("knee_angle_r")
        # ax.scatter(knee_l, knee_r, s=3, c=v_r)
        preds = model.predict(inputs)
        ax.scatter(self.phase, correct, s=3, c="gray", alpha=0.2)
        ax.scatter(self.phase, preds, s=3, c=correct, cmap="plasma", alpha=0.3)
        rmse = np.sqrt(np.mean(np.pow(correct - preds, 2)))
        print(f"{self.get_name()}: {rmse:.01}")


TRAINING_DATASETS = [
    Dataset(0.6, "BT02_normal_walk_1_2_0-6"),
    Dataset(0.6, "BT06_normal_walk_1_2_0-6"),
    Dataset(1.2, "BT02_normal_walk_1_2_1-2"),
    Dataset(1.2, "BT06_normal_walk_1_2_1-2"),
    Dataset(1.8, "BT02_normal_walk_1_2_1-8"),
    Dataset(1.8, "BT06_normal_walk_1_2_1-8"),
    Dataset(2.0, "BT02_normal_walk_1_4_2-0"),
    Dataset(2.0, "BT06_normal_walk_1_2_2-0"),
    Dataset(2.5, "BT02_normal_walk_1_4_2-5"),
    Dataset(2.5, "BT06_normal_walk_1_2_2-5"),
]

TESTING_DATASETS = [
    Dataset(1.2, "BT08_normal_walk_1_2_1-2"),
    Dataset(1.8, "BT08_normal_walk_1_2_1-8"),
    Dataset(2.0, "BT08_normal_walk_1_2_2-0"),
    Dataset(2.5, "BT08_normal_walk_1_2_2-5"),
]


def main():
    _, axes = plt.subplots(2, 2)
    inputs = np.array([])
    outputs = np.array([])
    first = True
    for dataset in TRAINING_DATASETS:
        if not first:
            d_inputs, d_outputs = dataset.make_training_data()
            inputs = np.vstack([inputs, d_inputs])
            outputs = np.vstack([outputs, d_outputs])
        else:
            inputs, outputs = dataset.make_training_data()
            first = False

    model = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=4, include_bias=False),
        Ridge(alpha=2.5),
    )
    model.fit(inputs, outputs)

    for dataset, ax in zip(TESTING_DATASETS, np.ravel(axes)):
        ax.set_title(dataset.get_name())
        dataset.graph(ax, model)

    # save model to disk
    joblib.dump(model, "model.xz", compress=True)


if __name__ == "__main__":
    main()
    plt.show()
