import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ANGLE_PATHS = [
    "BT01_normal_walk_3_0-6_off_angle_filt.csv",
    "BT01_normal_walk_3_1-2_off_angle_filt.csv",
    "BT01_normal_walk_3_1-8_off_angle_filt.csv",
    "BT01_normal_walk_3_2-0_off_angle_filt.csv",
    "BT01_normal_walk_3_2-5_off_angle_filt.csv",
]
MOMENT_PATHS = [
    "BT01_normal_walk_3_0-6_off_moment_filt.csv",
    "BT01_normal_walk_3_1-2_off_moment_filt.csv",
    "BT01_normal_walk_3_1-8_off_moment_filt.csv",
    "BT01_normal_walk_3_2-0_off_moment_filt.csv",
    "BT01_normal_walk_3_2-5_off_moment_filt.csv",
]
speeds = [0.6, 1.2, 1.8, 2.0, 2.5]


def phase_variable(angle, angular_rate, k):
    return (np.atan2(k * angular_rate, angle) / np.pi * 0.5) + 0.5


fig = plt.figure()
ax = fig.add_subplot(projection="3d")

x_values = []
y_values = []
z_values = []

for i, (angle_path, moment_path, speed) in enumerate(zip(ANGLE_PATHS, MOMENT_PATHS, speeds)):
    angle_df = pd.read_csv(
        "forward_walking_molinaro/" + angle_path, usecols=["time", "hip_flexion_l"]
    )
    moment_df = pd.read_csv(
        "Forward Walking Molinaro/" + moment_path,
        usecols=["time", "hip_flexion_l_moment"],
    )

    t = angle_df.get("time").to_numpy()
    angle = angle_df.get("hip_flexion_l").to_numpy()
    angular_rate = np.gradient(angle, t)
    moment = moment_df.get("hip_flexion_l_moment").to_numpy()
    pv = phase_variable(angle, angular_rate, 1)

    x_values.append(pv)
    y = np.full(pv.shape, speed)
    y_values.append(y)
    z_values.append(moment)

    coeffs = np.polyfit(pv, moment, 8)

    xx = np.linspace(0, 1, 100)
    zz = np.poly1d(coeffs)(xx)
    ax.plot(xx, np.full(xx.shape, speed), zz)
    # ax.scatter(pv, y, moment, s=3, color="gray")

deg_x = 8
deg_y = 4

x = np.concatenate(x_values)
y = np.concatenate(y_values)
z = np.concatenate(z_values)

powers = [(m, n) for m in range(deg_x) for n in range(deg_y)]
vandermonde = np.column_stack([x ** m * y ** n for m, n in powers])

coeffs, _, _, _ = np.linalg.lstsq(vandermonde, z)

x_grid = np.linspace(x.min(), x.max(), 200)
y_grid = np.linspace(y.min(), y.max(), 50)
xx, yy = np.meshgrid(x_grid, y_grid)

grid_v = np.column_stack([xx.ravel() ** m * yy.ravel() ** n for m,n in powers])
zz = (grid_v @ coeffs).reshape(xx.shape)

ax.plot_surface(xx, yy, zz, alpha=0.8, cmap="viridis") 

plt.show()


