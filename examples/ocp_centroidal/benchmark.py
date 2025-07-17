import matplotlib.pyplot as plt
import numpy as np


T_dmpc = [50, 40, 30, 25, 20]
T_fd = [50, 40, 30, 25, 20]

T_dmpc = np.array(T_dmpc)*1e-2
T_dmpc = list(T_dmpc)
T_fd = np.array(T_fd)*1e-2
T_fd = list(T_fd)

time_fd = [88.0, 83.5, 75.5, 72.6, 66] # temps calcul moyen d'une itération
time_dmpc = [62.2, 52.0, 41.6, 37.0, 33.1]

# seulemnt mpc_fd.iterate
# time_dmpc = [42.9, 34.2, 24.9, 20.6, 17.9]

#num threads = 1
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

# # add anotation for the potential point of fd at 0.2 70 saying inaccessible with a marker
# plt.annotate("Inaccessible", (0.2, 70), textcoords="offset points",
#              xytext=(0, 10), ha='center', fontsize=8, color="blue")
# # add one marker at 0.2 70
# plt.scatter(0.2, 70, color="blue", s=50, marker="x")


plt.xticks(T_dmpc + [0.05], [f"{t:.2f}" for t in T_dmpc] + ["0.05"])
plt.xlim(0.15, 0.55)
plt.ylim(0, 100)
plt.tight_layout()

plt.legend()
plt.show()


