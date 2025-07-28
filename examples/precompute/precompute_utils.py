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
