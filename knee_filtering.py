import sys
from functools import cache
import matplotlib.pyplot as plt
import numpy as np
import pandas
from scipy.signal import find_peaks
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

BASE_DIR = "forward_walking_molinaro"
TRAINING_DATASETS = [
    (1.2, "BT02_normal_walk_1_2_1-2"),
    (1.8, "BT02_normal_walk_1_2_1-8"),
    (2.0, "BT02_normal_walk_1_4_2-0"),
    (2.5, "BT02_normal_walk_1_4_2-5"),
]

TESTING_DATASETS = [
    (1.2, "BT06_normal_walk_1_2_1-2"),
    (1.8, "BT06_normal_walk_1_2_1-8"),
    (2.0, "BT06_normal_walk_1_2_2-0"),
    (2.5, "BT06_normal_walk_1_2_2-5"),
]

VALUES = ["knee_angle_l", "hip_adduction_l", "knee_angle_r", "hip_adduction_r"]


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
    def __init__(self, file_base, speed):
        self.name = file_base
        self.angle_df = pandas.read_csv(f"{BASE_DIR}/{file_base}_off_angle_filt.csv")
        self.moment_df = pandas.read_csv(f"{BASE_DIR}/{file_base}_off_moment_filt.csv")

        self.time = self.angle_df.get("time").to_numpy()
        self.time = self.time - self.time[0]  # start at 0
        self.dt = np.diff(self.time)
        self.speed = speed

        # processing time!

        angle, _, _ = self.load("knee_angle_l")

        norm = angle - np.mean(angle)
        acf = np.correlate(norm, norm, mode="full")[len(norm) - 1 :]
        peaks, _ = find_peaks(acf)
        period_ticks = max(peaks, key=lambda p: acf[p])
        self.period = period_ticks * np.mean(self.dt)

        cycles = np.resize(angle, (len(angle) // period_ticks, period_ticks))
        mean_cycle = np.mean(cycles, axis=0)

        peaks, _ = find_peaks(-mean_cycle)
        start = min(peaks, key=lambda p: mean_cycle[p])
        offset = start * np.mean(self.dt)
        shifted_time = self.time - offset
        self.phase = np.mod(shifted_time, self.period) / self.period

    def get_name(self):
        return f"{self.speed} meters per second"

    def load(self, name):
        x = self.angle_df.get(name).to_numpy()
        v = np.gradient(x, self.time)
        moment = self.moment_df.get(name + "_moment")
        if np.isnan(moment).any():
            print(f"[ERROR]: {self.name}.{name} has NaN moment")
            sys.exit(1)
        return (x, v, moment)

    def make_phase_func_input(self):
        angle, angular_rate, _ = self.load("knee_angle_l")
        flexion, flexion_rate, _ = self.load("hip_flexion_r")

        return np.column_stack(
            [
                angle,
                angular_rate / self.speed,
                flexion,
                flexion_rate / self.speed,
                self.speed * np.ones(self.time.shape),
            ]
        )

    def make_phase_func_output(self):
        phase_as_angle = 2 * np.pi * self.phase
        return np.column_stack([np.sin(phase_as_angle), np.cos(phase_as_angle)])

    def make_moment_func_input(self):
        return np.column_stack(
            [
                self.phase,
                np.ones(self.phase.shape) * self.speed,
            ]
        )

    def make_moment_func_output(self):
        _, _, moment = self.load("knee_angle_l")
        return np.column_stack([moment])

    def graph(self, ax, phase_model, moment_model, target="phase"):
        _, _, moment = self.load("knee_angle_l")

        phase_pred_trig = phase_model.predict(self.make_phase_func_input())
        phase_pred_angle = np.arctan2(phase_pred_trig[:, 0], phase_pred_trig[:, 1])
        phase_pred = np.mod(phase_pred_angle / (2 * np.pi), 1)
        if target == "phase":
            ax.scatter(self.phase, moment, s=3, c=phase_pred, alpha=0.4, cmap="plasma")
        if target == "moment":
            moment_pred = moment_model.predict(
                np.column_stack(
                    [
                        phase_pred,
                        np.ones(self.phase.shape) * self.speed,
                    ]
                )
            )
            # moment_pred = moment_model.predict(self.make_moment_func_input())
            ax.scatter(self.phase, moment, s=3, c="gray", alpha=0.2)
            ax.scatter(self.phase, moment_pred, s=3, c=moment, cmap="plasma", alpha=0.5)


def main():
    _, axes = plt.subplots(2, 2)
    phase_inputs = np.array([])
    phase_outputs = np.array([])
    moment_inputs = np.array([])
    moment_outputs = np.array([])
    first = True
    for d in TRAINING_DATASETS:
        speed, name = d
        dataset = Dataset(name, speed)
        if not first:
            phase_inputs = np.vstack([phase_inputs, dataset.make_phase_func_input()])
            phase_outputs = np.vstack([phase_outputs, dataset.make_phase_func_output()])
            moment_inputs = np.vstack([moment_inputs, dataset.make_moment_func_input()])
            moment_outputs = np.vstack(
                [moment_outputs, dataset.make_moment_func_output()]
            )
        else:
            phase_inputs = dataset.make_phase_func_input()
            phase_outputs = dataset.make_phase_func_output()
            moment_inputs = dataset.make_moment_func_input()
            moment_outputs = dataset.make_moment_func_output()
            first = False

    phase_model = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=4, include_bias=False),
        Ridge(alpha=2.5),
    )
    phase_model.fit(phase_inputs, phase_outputs)

    moment_model = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=7, include_bias=False),
        Ridge(alpha=0.5),
    )
    moment_model.fit(moment_inputs, moment_outputs)

    for d, ax in zip(TESTING_DATASETS, np.ravel(axes)):
        speed, name = d
        dataset = Dataset(name, speed)
        ax.set_title(dataset.get_name())
        dataset.graph(ax, phase_model, moment_model, "moment")


if __name__ == "__main__":
    main()
    plt.show()
