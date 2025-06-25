import matplotlib.pyplot as plt
import numpy as np
import ndcurves

def plot_results(res:np.array, T_jump:int=0, T_land:int=0, show_plot:bool=True, savefig:str=""):
    """ Create plots of com position, linear and angular momentum"""
    com = res[:,:3]
    lin_m = res[:,3:6]
    ang_m = res[:,6:]
    time = np.arange(len(com))

    # plot com positions x, y, z as a function of time, in subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    axs[0].plot(time, com[:, 0], label='x', color='blue')
    axs[0].plot(time, com[:,1], label="y", color='red')
    axs[0].plot(time, com[:,2], label="z", color='orange')
    axs[0].legend()
    axs[0].grid(True)

    # linear momentum
    axs[1].plot(time, lin_m[:,0], label="x", color="blue")
    axs[1].plot(time, lin_m[:,1], label="y", color="red")
    axs[1].plot(time, lin_m[:,2], label="z", color="orange")
    axs[1].legend()
    axs[1].grid(True)

    # angular momentum
    axs[2].plot(time, ang_m[:,0], label="x", color="blue")
    axs[2].plot(time, ang_m[:,1], label="y", color="red")
    axs[2].plot(time, ang_m[:,2], label="z", color="orange")
    axs[2].legend()
    axs[2].grid(True)

    axs[0].set_title("State of robot")
    axs[0].set_ylabel("COM")
    axs[1].set_ylabel("linear momentum")
    axs[2].set_ylabel("angular momentum")
    axs[2].set_xlabel("time (steps of simulation)")

    if T_jump >0:
        for i in (0,1,2):
            axs[i].axvline(T_jump, ls='--', color="teal")

    if T_land >0:
        for i in (0,1,2):
            axs[i].axvline(T_land, ls='--', color="teal")

    # add labels for the two minimums and the maximum of the com z position
    if len(com) > 0 and T_jump >0 and T_land >0:
        min1_z = np.min(com[:T_jump,2])
        min2_z = np.min(com[T_land:,2])
        max_z = np.max(com[:,2])
        min1_idx = np.argmin(com[:T_jump,2])
        min2_idx = np.argmin(com[T_land:,2]) + T_land
        max_idx = np.argmax(com[:,2])
        axs[0].annotate(f"min1 z: {min1_z:.2f} at {min1_idx}", 
                        xy=(min1_idx - 10, min1_z - 0.05))
        axs[0].annotate(f"min2 z: {min2_z:.2f} at {min2_idx}", 
                        xy=(min2_idx - 10, min2_z - 0.05))
        axs[0].annotate(f"max z: {max_z:.2f} at {max_idx}", 
                        xy=(max_idx + 2, max_z ))
        axs[0].scatter(min1_idx, min1_z, color='black', s=50, marker='x')
        axs[0].scatter(min2_idx, min2_z, color='black', s=50, marker='x')
        axs[0].scatter(max_idx, max_z, color='black', s=50, marker='x')

    plt.tight_layout()

    if len(savefig) > 0:
        plt.savefig(f"results/{savefig}.png")
    if show_plot:
        plt.show()

# poor function
def global_comp_predictions(com_c:np.array, com_fd:np.array, comp_times, show_plot:bool=True, save_png:str=""):
    n = com_c.shape[0]
    fig, axs = plt.subplots(3,1,figsize=(10,15), sharex=True)
    colors1 = ['red', 'blue', 'orange']
    colors2 = ['cyan', 'teal', 'lime']
    for i in range(n):
        com_c_i = com_c[i]
        com_fd_i = com_fd[i]
        time = np.arange(len(com_c_i)) + comp_times[i]
        axs[0].plot(time, com_c_i[:,0], label="Centroidal x", color=colors1[i%3], ls='--')
        axs[0].plot(time, com_fd_i[:,0], label="FD x", color=colors2[i%3])
        axs[1].plot(time, com_c_i[:,1], label="Centroidal x", color=colors1[i%3], ls='--')
        axs[1].plot(time, com_fd_i[:,1], label="FD x", color=colors2[i%3])
        axs[2].plot(time, com_c_i[:,2], label="Centroidal x", color=colors1[i%3], ls='--')
        axs[2].plot(time, com_fd_i[:,2], label="FD x", color=colors2[i%3])

    ylabels = ["x","y","z"]

    for j,y in enumerate(ylabels):
        axs[j].legend()
        axs[j].grid(True)
        axs[j].set_ylabel(f"COM {y}")

    axs[0].set_title("Comparison of Centroidal and Full Dynamics Trajectory")
    axs[2].set_xlabel("Time (steps of simulation)")

    plt.tight_layout()

    if len(save_png) > 0:
        plt.savefig(f"/home/matthieu/simple/simple-mpc_ws/src/simple-mpc/examples/results/{save_png}.png")
    if show_plot:
        plt.show()
    

def plot_forces(f:np.array, show_plot:bool=True, save_png:str=""):
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
    if show_plot:
        plt.show()


    

def compare_predictions(res_centroidal:np.array, res_fd:np.array, show_plot:bool=True, save_png:str=""):
    """Creates a plot comparing the centroidal trajectory from centroidal model and full dynamics model. 
    Same function than plot_results but with 2 lines in the same plot."""
    com_centroidal = res_centroidal[:,:3]
    com_fd = res_fd[:,:3]
    time = np.arange(len(com_centroidal))
    time2 = np.arange(len(com_fd))

    fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

    # plot com positions x, y, z as a function of time, in subplots
    axs[0].plot(time, com_centroidal[:, 0], label='Centroidal x', color='blue')
    axs[0].plot(time2, com_fd[:, 0], label='FD x', color='cyan')
    axs[0].plot(time, com_centroidal[:, 1], label='Centroidal y', color='red')
    axs[0].plot(time2, com_fd[:, 1], label='FD y', color='orange')
    axs[0].plot(time, com_centroidal[:, 2], label='Centroidal z', color='green')
    axs[0].plot(time2, com_fd[:, 2], label='FD z', color='lime')
    axs[0].legend()
    axs[0].grid(True)

    # linear momentum
    lin_m_centroidal = res_centroidal[:,3:6]
    lin_m_fd = res_fd[:,3:6]
    axs[1].plot(time, lin_m_centroidal[:,0], label="Centroidal x", color="blue")
    axs[1].plot(time2, lin_m_fd[:,0], label="FD x", color="cyan")
    axs[1].plot(time, lin_m_centroidal[:,1], label="Centroidal y", color="red")
    axs[1].plot(time2, lin_m_fd[:,1], label="FD y", color="orange")
    axs[1].plot(time, lin_m_centroidal[:,2], label="Centroidal z", color="green")
    axs[1].plot(time2, lin_m_fd[:,2], label="FD z", color="lime")
    axs[1].legend()
    axs[1].grid(True)

    # angular momentum
    ang_m_centroidal = res_centroidal[:,6:]
    ang_m_fd = res_fd[:,6:]
    axs[2].plot(time, ang_m_centroidal[:,0], label="Centroidal x", color="blue")
    axs[2].plot(time2, ang_m_fd[:,0], label="FD x", color="cyan")
    axs[2].plot(time, ang_m_centroidal[:,1], label="Centroidal y", color="red")
    axs[2].plot(time2, ang_m_fd[:,1], label="FD y", color="orange")
    axs[2].plot(time, ang_m_centroidal[:,2], label="Centroidal z", color="green")
    axs[2].plot(time2, ang_m_fd[:,2], label="FD z", color="lime")
    axs[2].legend()
    axs[2].grid(True)   
    axs[0].set_title("Comparison of Centroidal and Full Dynamics Trajectory")
    axs[0].set_ylabel("COM")
    axs[1].set_ylabel("Linear Momentum")
    axs[2].set_ylabel("Angular Momentum")
    axs[2].set_xlabel("Time (steps of simulation)")
    plt.tight_layout()

    if len(save_png) > 0:
        plt.savefig(f"results/{save_png}.png")
    if show_plot:
        plt.show()

# poor function
def global_comp_predictions(com_c:np.array, com_fd:np.array, comp_times, show_plot:bool=True, save_png:str=""):
    n = com_c.shape[0]
    fig, axs = plt.subplots(3,1,figsize=(10,15), sharex=True)
    colors1 = ['red', 'blue', 'orange']
    colors2 = ['cyan', 'teal', 'lime']
    for i in range(n):
        com_c_i = com_c[i]
        com_fd_i = com_fd[i]
        time = np.arange(len(com_c_i)) + comp_times[i]
        axs[0].plot(time, com_c_i[:,0], label="Centroidal x", color=colors1[i%3], ls='--')
        axs[0].plot(time, com_fd_i[:,0], label="FD x", color=colors2[i%3])
        axs[1].plot(time, com_c_i[:,1], label="Centroidal x", color=colors1[i%3], ls='--')
        axs[1].plot(time, com_fd_i[:,1], label="FD x", color=colors2[i%3])
        axs[2].plot(time, com_c_i[:,2], label="Centroidal x", color=colors1[i%3], ls='--')
        axs[2].plot(time, com_fd_i[:,2], label="FD x", color=colors2[i%3])

    ylabels = ["x","y","z"]

    for j,y in enumerate(ylabels):
        axs[j].legend()
        axs[j].grid(True)
        axs[j].set_ylabel(f"COM {y}")

    axs[0].set_title("Comparison of Centroidal and Full Dynamics Trajectory")
    axs[2].set_xlabel("Time (steps of simulation)")

    plt.tight_layout()

    if len(save_png) > 0:
        plt.savefig(f"/home/matthieu/simple/simple-mpc_ws/src/simple-mpc/examples/results/{save_png}.png")
    if show_plot:
        plt.show()
    

def plot_forces(f:np.array, show_plot:bool=True, save_png:str=""):
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
    if show_plot:
        plt.show()

def plot_com_3d(com:np.array, show_plot:bool=True, save_png:str=""):
    """ Creates a 3D plot of COM trajectory. """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    if len(com.shape) == 1:
        ax.scatter(com [0], com[1], com[2], marker="o", label="COM position", color="blue")
    else:
        # plot com position in 3D
        ax.plot(com[:, 0], com[:, 1], com[:, 2], ls='dashdot', label='COM Position', color='blue')

        # add start and end points for COM
        ax.scatter(com[0, 0], com[0, 1], com[0, 2], color='green', s=80, marker='o', label='COM Start')
        ax.scatter(com[-1, 0], com[-1, 1], com[-1, 2], color='black', s=80, marker='x', label='COM End')

        # add points every 10 steps
        for i in range(0, com.shape[0], 10):
            ax.scatter(com[i, 0], com[i, 1], com[i, 2], color='blue', s=20, marker='o')
    
    # add grid and labels
    ax.grid(True)
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title('Center of Mass (COM) Position in 3D')
    ax.legend()
    
    plt.tight_layout()

    if (len(save_png) > 0):
        plt.savefig(f"results/{save_png}.png")

    if show_plot:
        plt.show()

def create_contact_phases(c_phases:list, timings:list, n_cycles:int):
    """ create the contact phase for each simulation step."""
    contacts = []
    for n in range(n_cycles):
        for c,t in zip(c_phases, timings):
            contacts += [c] * t
    return contacts

def create_urefs(c_phases:list, timings:list, n_cycles:int, possible_u:dict):
    """ Creates the contact forces (urefs) of reference according to the sequence of contacts
        for each simulation step.
        urefs are created by linear interpolation of the known values"""
    uref = []
    for _ in range(n_cycles):
        n = len(c_phases)
        for i in range(n):
            u0 = possible_u[c_phases[i%n]]
            u1 = possible_u[c_phases[(i+1)%n]]
            T = timings[i]
            for t in range(T):
                uref.append((u1*t + (T - t)*u0)/T)
    return uref


def shapeState(q_current, v_current, nq, nxq, cj_ids):
    """ get the full dynamics state of the robot"""
    x_internal = np.zeros(nxq)
    x_internal[:7] = q_current[:7]
    x_internal[nq:nq + 6] = v_current[:6]
    i = 0
    for jointID in cj_ids:
        if jointID > 1:
            x_internal[i + 7] = q_current[jointID + 5]
            x_internal[nq + i + 6] = v_current[jointID + 4]
            i += 1
    
    return x_internal

# WORK IN PROGRESS
class footTrajectory:
    def __init__(
        self, start_pose_FL, start_pose_FR, start_pose_RL, start_pose_RR, 
        T_ss, T_ds, nsteps, swing_apex
    ):
        self.start_poses = [start_pose_FL, start_pose_FR, start_pose_RL, start_pose_RR]
        self.final_poses = [start_pose_FL, start_pose_FR, start_pose_RL, start_pose_RR]

        self.T_ds = T_ds
        self.T_ss = T_ss
        self.nsteps = nsteps
        self.swing_apex = swing_apex
    
    def updateForward(self, x_f_left, x_f_right, y_gap, y_forward, z_height_left, z_height_right, swing_apex):
        self.translationRight = np.array([x_f_right, -y_gap - y_forward, z_height_right])
        self.translationLeft = np.array([x_f_left, y_gap, z_height_left])
        self.swing_apex = swing_apex
    
    def updateTrajectory(self, takeoff, land, feet_poses):
        """ update feet trajectories"""
        i=0
        for l in land:
            if l<0:
                self.start_poses[i] = feet_poses[i]
                self.final_poses[i] = feet_poses[i]
            i+=1
        
        i=0
        # pour le moment, saut sur place
        for t in takeoff:
            if t < self.T_ds and t >=0:
                self.start_poses[i] = feet_poses[i]
                self.start_poses[i] = feet_poses[i]     
            i+=1
        
        swing_trajectories = []
        feet_refs = []
        for i in range(4):
            swing_trajectories.append(self.defineBezier(self.swing_apex, 0, 1, self.start_poses[i], self.final_poses[i]))
            if land[i] > -1:
                feet_refs.append(self.foot_trajectory(self.nsteps, 
                                                       land[i],
                                                       self.start_poses[i], 
                                                       self.final_poses[i], 
                                                       swing_trajectories[i], 
                                                       self.T_ss))
            else:
                feet_refs.append([self.start_poses[i] for _ in range(self.nsteps)])

        return feet_refs


    def defineBezier(self, height, time_init, time_final, placement_init, placement_final):
        wps = np.zeros([3, 9])
        for i in range(4):  # init position. init vel,acc and jerk == 0
            wps[:, i] = placement_init.translation
        # compute mid point (average and offset along z)
        wps[:, 4] = placement_init.translation * 3/4 + placement_final.translation * 1/4
        wps[2, 4] += height
        for i in range(5, 9):  # final position. final vel,acc and jerk == 0
            wps[:, i] = placement_final.translation
        translation = ndcurves.bezier3(wps, time_init, time_final)
        pBezier = ndcurves.piecewise_SE3(
            ndcurves.SE3Curve(
                translation, placement_init.rotation, placement_final.rotation
            )
        )
        return pBezier
    
    def foot_trajectory(self, T, time_to_land, initial_pose, final_pose, trajectory_swing, TsingleSupport):
        placement = []
        for t in range(
            time_to_land, time_to_land - T, -1
        ):
            if t <= 0:
                placement.append(final_pose)
            elif t > TsingleSupport:
                placement.append(initial_pose)
            else:
                swing_pose = initial_pose.copy()
                swing_pose.translation = trajectory_swing.translation(
                    float(TsingleSupport - t) / float(TsingleSupport)
                )
                swing_pose.rotation = trajectory_swing.rotation(
                    float(TsingleSupport - t) / float(TsingleSupport)
                )
                placement.append(swing_pose)

        return placement
    
    def yawRotation(self, yaw):
        Ro = np.array(
            [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
        )
        return Ro
    
def scan_list(list_Fs):
    for i in range(len(list_Fs)):
        list_Fs[i] -= 1
    if len(list_Fs) > 0 and list_Fs[0] == -1:
        list_Fs.remove(list_Fs[0])

def update_timings(landings, takeoffs):
    land = [-1, -1, -1, -1]
    takeoff = [-1, -1, -1, -1]
    i = 0

    for l in landings:
        scan_list(l)
        if len(l) > 0:
            land[i] = l[0]
        i += 1
    i = 0

    for t in takeoffs:
        scan_list(t)
        if len(t)>0:
            takeoff[i] = t[0]
        i += 1

    return land, takeoff

if __name__ == "__main__":
    plot_results(np.random.randint(1,20,(9,20)))