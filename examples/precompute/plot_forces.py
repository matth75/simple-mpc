import numpy as np

from precompute_utils import plot_forces

with open("fd_traj.npy", "rb") as f:
    xrefs = np.load(f)
    urefs = np.load(f)
    forces = np.load(f)


plot_forces(forces, T_0=99, T_ds=150)