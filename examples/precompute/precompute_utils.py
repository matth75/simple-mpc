import matplotlib.pyplot as plt
import numpy as np


def create_contact_phases(c_phases:list, timings:list, n_cycles:int):
    """ create the contact phase for each simulation step."""
    contacts = []
    for n in range(n_cycles):
        for c,t in zip(c_phases, timings):
            contacts += [c] * t
    return contacts

def plot_torques(us):
    # plot the torques of each joint from each 4 legs of the robot using us which is the output of the OCP on 4 subplots
    plt.figure(figsize=(10, 8))
    for i in range(4):
        plt.subplot(4, 1, i + 1)
        plt.plot(us[:, i * 3], label='joint 1')
        plt.plot(us[:, i * 3 + 1], label='joint 2')
        plt.plot(us[:, i * 3 + 2], label='joint 3')
        plt.title(f'Leg {i + 1} torques')
        plt.xlabel('Time Step')
        plt.ylabel('Torque (N.m)')
        plt.grid()
        plt.legend()
    plt.tight_layout()
    plt.show()
    

def plot_forces(f:np.array, T_0:int, T_ds:int, show_plot:bool=True, save_png:str=""):
    fig, axs = plt.subplots(4, 1, figsize=(15,10), sharex=True)
    time = np.arange(f.shape[0])

    labels = ["FL foot", "FR_foot", "RL foot", "RR foot"]

    for i,l in enumerate(labels):
        axs[i].plot(time, f[:,i*3], label = 'x')
        axs[i].plot(time, f[:,i*3 + 1], label = 'y')
        axs[i].plot(time, f[:,i*3 + 2], label = 'z')
        axs[i].set_ylabel(l)
        axs[i].legend()
        axs[i].grid(True)
    
    axs[0].set_title("Forces on each leg")
    axs[-1].set_xlabel("Time (steps of simulation)")
    plt.tight_layout()
    
    if len(save_png) > 0:
        plt.savefig(f"/home/matthieu/simple/simple-mpc_ws/src/simple-mpc/examples/results/{save_png}.png")

    # add markers with value of forces z component at T_ds
    if T_ds > 0 and T_ds < f.shape[0]:
        for i in range(4):
            axs[i].scatter(T_ds, f[T_ds, i*3 + 2], color='black', s=50, marker='x')
            axs[i].annotate(f"{f[T_ds, i*3 + 2]:.2f}", 
                            xy=(T_ds + 2, f[T_ds, i*3 + 2] - 10), 
                            fontsize=10, color='black')
            
    # add markers with value of forces z component at T_0
    if T_0 > 0 and T_0 < f.shape[0]:
        for i in range(4):
            axs[i].scatter(T_0, f[T_0, i*3 + 2], color='black', s=50, marker='x')
            axs[i].annotate(f"{f[T_0, i*3 + 2]:.2f}", 
                            xy=(T_0 + 2, f[T_0, i*3 + 2] - 10), 
                            fontsize=10, color='black')
                  
    if show_plot:
        plt.show()


#TODO: Generate cycle horizon from data !!