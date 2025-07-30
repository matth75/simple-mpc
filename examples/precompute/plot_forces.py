import numpy as np
import matplotlib.pyplot as plt

from precompute_utils import plot_forces, plot_torques

# with open("fd_traj.npy", "rb") as f:
#     xrefs = np.load(f)
#     urefs = np.load(f)
#     forces = np.load(f)

data = np.load("fd_trajs_from_ocp.npz")
forces = data["forces"]

torques = data["us"]

x = data["xs"]
time = np.arange(np.shape(x)[0])
plt.plot(time, x[:,2])
plt.grid(True)
plt.xlabel("time steps")
plt.ylabel("COM z (m)")
plt.title("Evolution of COM z component")
plt.show()

plot_torques(torques)
plot_forces(forces, T_0=99, T_ds=150)