""" 
Plot and compare compute time of Double MPC scheme and Full Dynamics MPC scheme.

Time measured is before mpc_centroidal.iterate() and after mpc_fd.iterate() for the double MPC scheme.
Time measured is before and after mpc_fd.iterate() for the FD MPC scheme.

The solver settings are the same for both schemes :
num_threads = 1, TOL = 1e-4, mu_init = 1e-8...
 """

import matplotlib.pyplot as plt
import numpy as np


T_dmpc = [50, 40, 30, 25, 20]
T_fd = [50, 40, 30, 25, 20]

T_dmpc = np.array(T_dmpc)*1e-2
T_dmpc = list(T_dmpc)
T_fd = np.array(T_fd)*1e-2
T_fd = list(T_fd)

# -------- Comparison of compute times ------------ #

time_dmpc = [62.2, 52.0, 41.6, 37.0, 33.1]

time_fd = [47.2, 36.0, 29.0, 24.4, 17.6]

plt.plot(T_fd, time_fd, color="blue", label="Full Dynamics")
plt.plot(T_dmpc, time_dmpc, color="red", label="Double MPC")
plt.scatter(T_fd, time_fd, color="black", s=50, marker="x")
plt.scatter(T_dmpc, time_dmpc, color="black", s=50, marker="x")
plt.grid(True)
plt.ylabel("temps calcul moyen par itération (ms)")
plt.xlabel("longueur horizon (s)")
plt.title("Temps de calcul moyens des modèles FD et double MPC - saut de 0,3s")

# values of first and last elements of dmpc and fd anotated in the plot
plt.annotate(f"{time_dmpc[0]:.1f} ms", (T_dmpc[0], time_dmpc[0]), textcoords="offset points",
             xytext=(-10, 15), ha='center', fontsize=8, color="red")
plt.annotate(f"{time_dmpc[-1]:.1f} ms", (T_dmpc[-1], time_dmpc[-1]), textcoords="offset points",
             xytext=(-10, -15), ha='center', fontsize=8, color="red")
plt.annotate(f"{time_fd[0]:.1f} ms", (T_fd[0], time_fd[0]), textcoords="offset points",
             xytext=(-10, 10), ha='center', fontsize=8, color="blue")
plt.annotate(f"{time_fd[-1]:.1f} ms", (T_fd[-1], time_fd    [-1]), textcoords="offset points",
             xytext=(-10, -15), ha='center', fontsize=8, color="blue")

plt.xticks(T_dmpc + [0.05], [f"{t:.2f}" for t in T_dmpc] + ["0.05"])
plt.xlim(0.15, 0.55)
plt.ylim(0, 100)
plt.tight_layout()

plt.legend()
plt.show()

# ----------- comparison of jump lenghts ------------- #

max_Tss_fd = [0.4, 0.37, 0.33, 0.32, 0.3]
max_Tss_dmpc = [0.47, 0.45, 0.36, 0.35, 0.34]


plt.plot(T_fd, max_Tss_fd, color="blue", label="Full Dynamics")
plt.plot(T_dmpc, max_Tss_dmpc, color="red", label="Double MPC")
plt.scatter(T_fd, max_Tss_fd, color="black", s=50, marker="x")
plt.scatter(T_dmpc, max_Tss_dmpc, color="black", s=50, marker="x")
plt.grid(True)
plt.ylabel("durée maximale d'un saut (s)")
plt.xlabel("longueur horizon (s)")
plt.title("Comparaison durée maximale saut - longueur d'horizon")

# values of first and last elements of dmpc and fd anotated in the plot
plt.annotate(f"{max_Tss_dmpc[0]:.2f} s", (T_dmpc[0], max_Tss_dmpc[0]), textcoords="offset points",
             xytext=(-10, 15), ha='center', fontsize=8, color="red")
plt.annotate(f"{max_Tss_dmpc[-1]:.2f} s", (T_dmpc[-1], max_Tss_dmpc[-1]), textcoords="offset points",
             xytext=(-10, -15), ha='center', fontsize=8, color="red")
plt.annotate(f"{max_Tss_fd[0]:.2f} s", (T_fd[0], max_Tss_fd[0]), textcoords="offset points",
             xytext=(-10, 10), ha='center', fontsize=8, color="blue")
plt.annotate(f"{max_Tss_fd[-1]:.2f} s", (T_fd[-1], max_Tss_fd[-1]), textcoords="offset points",
             xytext=(-10, -15), ha='center', fontsize=8, color="blue")

plt.xticks(T_dmpc + [0.05], [f"{t:.2f}" for t in T_dmpc] + ["0.05"])
plt.xlim(0.15, 0.55)
plt.ylim(0.1, 0.6)
plt.tight_layout()

plt.legend()
plt.show()