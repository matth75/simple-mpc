import numpy as np
import matplotlib.pyplot as plt



with open("examples/com.npy", 'rb') as f:
    com = np.load(f)
    RR = np.load(f)
    RL = np.load(f)

with open("examples/forces.npy", 'rb') as f:
    force_FL = np.load(f)
    force_FR = np.load(f)
    force_RL = np.load(f)
    force_RR = np.load(f)

# plot com position in xy plane
def plot_com_xy():
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # simulation step of 1 ms
    time = np.arange(com.shape[0])
    
    # plot com position in xy plane
    ax.plot(com[:, 0], com[:, 1], label='COM Position', color='blue')
    ax.plot(RR[:,0], RR[:,1], label="RR foot pos", color='red')
    ax.plot(RL[:,0], RL[:,1], label="RL foot pos")

    # add grid and labels
    ax.grid(True)
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('Center of Mass (COM) Position in XY Plane')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig("examples/results/com_position.png")
    plt.show()

# 3D plot of com position
def plot_com_3d():
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # plot com position in 3D
    ax.plot(com[:, 0], com[:, 1], com[:, 2], ls='dashdot', label='COM Position', color='blue')
    ax.plot(RR[:,0], RR[:,1], RR[:,2], ls='--', label="RR foot pos", color='red')
    ax.plot(RL[:,0], RL[:,1], RL[:,2], ls='--', label="RL foot pos")

    # add start and end points for COM
    ax.scatter(com[0, 0], com[0, 1], com[0, 2], color='green', s=80, marker='o', label='COM Start')
    ax.scatter(com[-1, 0], com[-1, 1], com[-1, 2], color='black', s=80, marker='x', label='COM End')
    
    # add start and end points for feet
    ax.scatter(RR[0, 0], RR[0, 1], RR[0, 2], color='orange', s=80, marker='o', label='RR Start')
    ax.scatter(RR[-1, 0], RR[-1, 1], RR[-1, 2], color='red', s=80, marker='x', label='RR End')
    ax.scatter(RL[0, 0], RL[0, 1], RL[0, 2], color='purple', s=80, marker='o', label='RL Start')
    ax.scatter(RL[-1, 0], RL[-1, 1], RL[-1, 2], color='magenta', s=80, marker='x', label='RL End')

    # add points every 10 steps
    for i in range(0, com.shape[0], 10):
        ax.scatter(com[i, 0], com[i, 1], com[i, 2], color='blue', s=20, marker='o')
        ax.scatter(RR[i, 0], RR[i, 1], RR[i, 2], color='red', s=20, marker='o')
        ax.scatter(RL[i, 0], RL[i, 1], RL[i, 2], color='purple', s=20, marker='o')
    
    # add grid and labels
    ax.grid(True)
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title('Center of Mass (COM) Position in 3D')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig("examples/results/com_position_3d.png")
    plt.show()

def plot_forces_with_bounds( force_limit=100):
    time = np.arange(force_FL.shape[0])
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    feet = ['FL', 'FR', 'RL', 'RR']
    forces = [force_FL, force_FR, force_RL, force_RR]
    for i, (ax, f, name) in enumerate(zip(axs, forces, feet)):
        ax.plot(time, f[:, 0], label=f'{name} Fx')
        ax.plot(time, f[:, 1], label=f'{name} Fy')
        ax.plot(time, f[:, 2], label=f'{name} Fz')
        # ax.axhline(force_limit, color='r', linestyle='--', label='Force limit')
        # ax.axhline(-force_limit, color='r', linestyle='--')
        ax.set_ylabel(f'{name} Force [N]')
        ax.legend()
        ax.grid(True)
    axs[-1].set_xlabel('Time step')
    plt.tight_layout()
    plt.show()

with open('examples/qptorques.npy', 'rb') as f:
    torques = np.load(f)
    torques_limits = np.load(f)
    # torques_beforeQP = np.load(f)

n_joints = 12
n_legs = 4
n_steps = 100

T_ds = 30
T_lift = 10
T_land = 2
T_ss = 30

T = (T_ds/2, T_lift, T_ss, T_land, T_ds/2, T_ds/2, T_lift, T_ss, T_land, T_ds/2)
c0 = np.zeros(len(T) + 1)
for i in range(len(T)):
    c0[i] = sum(T[:i])

c0 = 500 + 10*c0

contact_changes = np.zeros(torques.shape[0])

joint_names = ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FR_hip_joint", "FR_thigh_joint",
                "FR_calf_joint", "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"]

def plot_torques_with_bounds():
    # TODO: plot torques for each leg in a seperate subfig
    fig, axs = plt.subplots(n_legs,1, figsize=(12,10), sharex=True)    # 4 legs

    # simulation step of 1 ms
    time = np.arange(torques.shape[0])

    # joint torques colors
    colors = ["orange", "red", "blue"]

    for t, ax in enumerate(axs):  # tous les couples
        for i in c0:
            ax.axvline(i, color='teal')     # steps of jumps...

        for j, c in enumerate(colors):
            t1 = [x[3*t + j] for x in torques]
            ax.plot(time, t1, label=f"{joint_names[3*t + j]}", color=c)
            ax.axhline(torques_limits[3*t+j], color=c, linestyle='--')
            ax.axhline(-torques_limits[3*t+j], color=c, linestyle='--')

        ax.set_ylabel("Torque [N.m]")

        ax.legend()
        ax.grid(True)
    axs[-1].set_xlabel('Time (ms)')
    plt.suptitle(r"Joint torques per leg with bounds ($\tau$)", fontsize=16)
    plt.tight_layout()
    plt.savefig("examples/results/torques_qp.png")
    plt.show()


# plot_com_3d()
plot_torques_with_bounds()
plot_forces_with_bounds()