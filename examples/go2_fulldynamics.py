import numpy as np
from bullet_robot import BulletRobot
from simple_mpc import (
    RobotModelHandler,
    RobotDataHandler,
    FullDynamicsOCP,
    MPC,
    IDSolver,
    Interpolator,
    FrictionCompensation
)
import example_robot_data as erd
import pinocchio as pin
import time
from utils import extract_forces
import copy

from ocp_centroidal.go2_utils import (create_contact_phases,
                                        plot_results,
                                        compare_predictions, 
                                        global_comp_predictions,
                                        plot_forces)

from ocp_centroidal.go2centrOPC import Go2CentroidalOCP

import matplotlib.pyplot as plt

# ####### CONFIGURATION  ############
# Load robot
URDF_SUBPATH = "/go2_description/urdf/go2.urdf"
base_joint_name ="root_joint"
robot_wrapper = erd.load('go2')

# Create Model and Data handler
model_handler = RobotModelHandler(robot_wrapper.model, "standing", base_joint_name)
model_handler.addFoot("FL_foot", base_joint_name, pin.XYZQUATToSE3(np.array([ 0.17, 0.15, 0.0, 0,0,0,1])))
model_handler.addFoot("FR_foot", base_joint_name, pin.XYZQUATToSE3(np.array([ 0.17,-0.15, 0.0, 0,0,0,1])))
model_handler.addFoot("RL_foot", base_joint_name, pin.XYZQUATToSE3(np.array([-0.24, 0.15, 0.0, 0,0,0,1])))
model_handler.addFoot("RR_foot", base_joint_name, pin.XYZQUATToSE3(np.array([-0.24,-0.15, 0.0, 0,0,0,1])))
data_handler = RobotDataHandler(model_handler)

nq = model_handler.getModel().nq
nv = model_handler.getModel().nv
nu = nv - 6
force_size = 3
nk = len(model_handler.getFeetNames())
nf = force_size

gravity = np.array([0, 0, -9.81])
fref = np.zeros(force_size)
fref[2] = -model_handler.getMass() / nk * gravity[2]
u0 = np.zeros(model_handler.getModel().nv - 6)

# full dynamics rcost weights
w_basepos = [0, 0, 0, 0, 0, 0]
w_legpos = [10, 10, 10]

w_basevel = [10, 10, 10, 10, 10, 10]
w_legvel = [0.1, 0.1, 0.1]
w_x = np.array(w_basepos + w_legpos * 4 + w_basevel + w_legvel * 4)
w_cent_lin = np.array([0.001, 0.001, 1])
w_cent_ang = np.array([0.0001, 0.0001, 0.001])
w_forces_lin = np.array([0.00001, 0.00001, 0.0001])
w_frame = np.eye(3)*1e3
w_com_fd = np.diag(np.array([0.01, 0.01, 1]))

dt = 0.01
problem_conf = dict(
    timestep=dt,
    w_x=np.diag(w_x),
    w_u=np.eye(u0.size) * 1e-4,
    w_cent=np.diag(np.concatenate((w_cent_lin, w_cent_ang))),
    gravity=gravity,
    force_size=3,
    w_forces=np.diag(w_forces_lin),
    w_frame=w_frame,
    w_com = w_com_fd,
    umin=-model_handler.getModel().effortLimit[6:],
    umax=model_handler.getModel().effortLimit[6:],
    qmin=model_handler.getModel().lowerPositionLimit[7:],
    qmax=model_handler.getModel().upperPositionLimit[7:],
    Kp_correction=np.array([0, 0, 20]),
    Kd_correction=np.array([100, 100, 100]),
    mu=0.8,
    Lfoot=0.01,
    Wfoot=0.01,
    torque_limits=True,
    kinematics_limits=True,
    force_cone=False,
    land_cstr=True
)
T = 50

dynproblem = FullDynamicsOCP(problem_conf, model_handler)
dynproblem.createProblem(model_handler.getReferenceState(), T, force_size, gravity[2], False)

T_ds = 50
T_lift = 20
T_land = 2
T_ss = 30
N_simu = int(0.01 / 0.001)
mpc_conf = dict(
    support_force=-model_handler.getMass() * gravity[2],
    TOL=1e-4,
    mu_init=1e-8,
    max_iters=1,
    num_threads=1,
    swing_apex=0.15, 
    T_fly=T_ss,
    T_contact=T_ds,
    timestep=dt,
)

mpc = MPC(mpc_conf, dynproblem)

""" Define contact sequence throughout horizon"""
contact_phase_quadru = {
    "FL_foot": True,
    "FR_foot": True,
    "RL_foot": True,
    "RR_foot": True,
}
contact_phase_lift_FL = {
    "FL_foot": False,
    "FR_foot": True,
    "RL_foot": True,
    "RR_foot": False,
}
contact_phase_lift_FR = {
    "FL_foot": True,
    "FR_foot": False,
    "RL_foot": False,
    "RR_foot": True,
}
contact_phase_lift = {
    "FL_foot": False,
    "FR_foot": False,
    "RL_foot": False,
    "RR_foot": False,
}

contact_phase_lift_Front = {
    "FL_foot": False,
    "FR_foot": False,
    "RL_foot": True,
    "RR_foot": True,
}

possible_contacts = {"stand":contact_phase_quadru,
                     "FL_up":contact_phase_lift_Front,    # Front legs up
                     "air":contact_phase_lift # jumping
                    }

c_phases = ["stand", "air", "stand"]

timings = [50, 30, 50]
cycles = 1  # number of repetitions of the sequence

# get the contacts
contact_phases = [possible_contacts[c] for c in create_contact_phases(c_phases, timings, cycles)]
contact_phasesOCP = [list(possible_contacts[c].values()) for c in create_contact_phases(c_phases, timings, cycles)]

mpc.generateCycleHorizon(contact_phases)

""" Initialize whole-body inverse dynamics QP"""
contact_ids = model_handler.getFeetIds()
id_conf = dict(
    contact_ids=contact_ids,
    x0=model_handler.getReferenceState(),
    mu=0.8,
    Lfoot=0.01,
    Wfoot=0.01,
    force_size=3,
    kd=0,
    w_force=0,
    w_acc=0,
    w_tau=1,
    verbose=False,
)

qp = IDSolver(id_conf, model_handler.getModel())

""" Friction """
fcompensation = FrictionCompensation(model_handler.getModel(), True)
""" Interpolation """
interpolator = Interpolator(model_handler.getModel())

""" Initialize simulation"""
device = BulletRobot(
    model_handler.getModel().names,
    erd.getModelPath(URDF_SUBPATH),
    URDF_SUBPATH,
    1e-3,
    model_handler.getModel(),
    model_handler.getReferenceState()[:3],
)

device.initializeJoints(model_handler.getReferenceState()[:nq])

for i in range(40):
    device.setFrictionCoefficients(i, 10, 0)
# device.changeCamera(1.0, 60, -15, [0.6, -0.2, 0.5])

device.changeCamera(1.0, 0, -15, [0, -0.5, 0.5])

q_meas, v_meas = device.measureState()
x_measured  = np.concatenate([q_meas, v_meas])
mpc.getDataHandler().updateInternalData(x_measured, False)

ref_foot_pose = [mpc.getDataHandler().getRefFootPose(mpc.getModelHandler().getFeetNames()[i]) for i in range(4)]
for pose in ref_foot_pose:
    pose.translation[2] = 0
device.showQuadrupedFeet(*ref_foot_pose)
Tmpc = len(contact_phases)

force_FL = []
force_FR = []
force_RL = []
force_RR = []
FL_measured = []
FR_measured = []
RL_measured = []
RR_measured = []
FL_references = []
FR_references = []
RL_references = []
RR_references = []
x_multibody = []
u_multibody = []
com_measured = []
solve_time = []
L_measured = []

torques = []
torques_before_qp = []

com_c =  []
com_fd = []
comp_times = [] # [60, 100, 110]

# vitesse du robot
v = np.zeros(6)
v[0] = 0
mpc.velocity_base = v

""" Test OCP """
go2centr = Go2CentroidalOCP(model_handler)

# number of simulation steps
n_steps = 200

for t in range(n_steps):
    print("Time " + str(t))
    land_LF = mpc.getFootLandCycle("FL_foot")
    land_RF = mpc.getFootLandCycle("RL_foot")
    takeoff_LF = mpc.getFootTakeoffCycle("FL_foot")
    takeoff_RF = mpc.getFootTakeoffCycle("RL_foot")
    """ print(
        "takeoff_RF = " + str(takeoff_RF) + ", landing_RF = ",
        str(land_RF) + ", takeoff_LF = " + str(takeoff_LF) + ", landing_LF = ",
        str(land_LF),
    ) """
    # if t == 70:
    #     for s in range(T):
    #         device.resetState(mpc.xs[s][:nq])
    #         #device.resetState(state_ref[s])
    #         time.sleep(0.05)
    #         print("s = " + str(s))
    #     exit()  

    device.moveQuadrupedFeet(
        mpc.getReferencePose(0, "FL_foot").translation,
        mpc.getReferencePose(0, "FR_foot").translation,
        mpc.getReferencePose(0, "RL_foot").translation,
        mpc.getReferencePose(0, "RR_foot").translation,
    )

    start = time.time()
    mpc.iterate(x_measured)
    end = time.time()
    solve_time.append(end - start)

    a0 = mpc.getStateDerivative(0)[nv:]
    a1 = mpc.getStateDerivative(1)[nv:]

    forces_vec0 = mpc.getContactForces(0)
    forces_vec1 = mpc.getContactForces(1)
    contact_states = mpc.ocp_handler.getContactState(0)

    # print(f'{t} : {list(contact_states)}')

    force_FL.append(forces_vec0[:3])
    force_FR.append(forces_vec0[3:6])
    force_RL.append(forces_vec0[6:9])
    force_RR.append(forces_vec0[9:12])

    forces = [forces_vec0, forces_vec1]
    ddqs = [a0, a1]
    xss = [mpc.xs[0], mpc.xs[1]]
    uss = [mpc.us[0], mpc.us[1]]

    # FL_measured.append(mpc.getDataHandler().getFootPose("FL_foot").translation)
    # FR_measured.append(mpc.getDataHandler().getFootPose("FR_foot").translation)
    # RL_measured.append(mpc.getDataHandler().getFootPose("RL_foot").translation)
    # RR_measured.append(mpc.getDataHandler().getFootPose("RR_foot").translation)
    # FL_references.append(mpc.getReferencePose(0, "FL_foot").translation)
    # FR_references.append(mpc.getReferencePose(0, "FR_foot").translation)
    # RL_references.append(mpc.getReferencePose(0, "RL_foot").translation)
    # RR_references.append(mpc.getReferencePose(0, "RR_foot").translation)
    # com_measured.append(mpc.getDataHandler().getData().com[0].copy())
    # L_measured.append(mpc.getDataHandler().getData().hg.angular.copy())


    if t in comp_times: # beginning of the jump : first [False, False, False ,False]
        contact_states = mpc.ocp_handler.getContactState(0)
        # x = mpc_fd.xs[0]
        print(mpc.solver.results)
        # data_handler.updateInternalData(x, False)
        # x_centr = data_handler.getCentroidalState()

        traj_fd = []
        forces_fd = []
        for s in range(T):  # len(mpc.xs) = 51, T = 50
            x_fd = mpc.xs[s]
            data_handler.updateInternalData(x_fd, False)
            traj_fd.append(data_handler.getCentroidalState())
            forces_fd.append(mpc.getContactForces(s))

        traj_fd = np.array(traj_fd)
        plot_results(traj_fd)


    for j in range(N_simu):
        # time.sleep(0.01)
        delay = j / float(N_simu) * dt

        x_interp = interpolator.interpolateState(delay, dt, xss)
        u_interp = interpolator.interpolateLinear(delay, dt, uss)
        # acc_interp = interpolator.interpolateLinear(delay, dt, ddqs)
        # force_interp = interpolator.interpolateLinear(delay, dt, forces)

        q_meas, v_meas = device.measureState()
        x_measured = np.concatenate([q_meas, v_meas])

        mpc.getDataHandler().updateInternalData(x_measured, True)

        current_torque = u_interp - 1. * mpc.Ks[0] @ model_handler.difference(
            x_measured, x_interp
        )

        # qp.solveQP(
        #     mpc.getDataHandler().getData(),
        #     contact_states,
        #     x_measured[nq:],
        #     acc_interp,
        #     current_torque,
        #     force_interp,
        #     mpc.getDataHandler().getData().M,
        # )

        # torques_before_qp.append(current_torque)

        # actually useless
        # qp_torque = qp.solved_torque.copy()

        # torques.append(qp_torque)

        # not needed, just need some right Ricatti gains
        # friction_torque = fcompensation.computeFriction(x_interp[nq + 6:], qp_torque)
        device.execute(current_torque)

        # u_multibody.append(copy.deepcopy(current_torque))
        # x_multibody.append(x_measured)

# global_comp_predictions(np.array(com_c), np.array(com_fd), comp_times, True, "mult_comp_predictions")

force_FL = np.array(force_FL)
force_FR = np.array(force_FR)
force_RL = np.array(force_RL)
force_RR = np.array(force_RR)
solve_time = np.array(solve_time)
FL_measured = np.array(FL_measured)
FR_measured = np.array(FR_measured)
RL_measured = np.array(RL_measured)
RR_measured = np.array(RR_measured)
FL_references = np.array(FL_references)
FR_references = np.array(FR_references)
RL_references = np.array(RL_references)
RR_references = np.array(RR_references)
com_measured = np.array(com_measured)
L_measured = np.array(L_measured)

torques = np.array(torques)

# centroidal residual

with open("examples/forces.npy", "wb") as f:
    np.save(f, force_FL)
    np.save(f, force_FR)
    np.save(f, force_RL)
    np.save(f, force_RR)

n_joints = 12
n_legs = 4

torques = np.array(torques)
torques_limits = np.array(model_handler.getModel().effortLimit[6:])

with open("examples/qptorques.npy", "wb") as f:
    np.save(f, torques)
    np.save(f, torques_limits)

time = np.arange(torques.shape[0])
tau1 = np.array([t[0] for t in torques])
tau2 = np.array([t[0] for t in torques_before_qp])


# plt.plot(time,tau2-tau1, label="abs(diff)")
# plt.grid(True)
# plt.legend()
# plt.title("différence des torques avant et après QP")
# plt.savefig("examples/diff_tau.png")
# plt.show()

# fig, axs = plt.subplots(2,1,figsize=(12,10), sharex=True)
# axs[0].plot(time, tau2, label="tau before QP")
# axs[1].plot(time, tau1, label="tau after QP")
# limite = torques_limits[0]
# axs[0].axhline(limite, linestyle="--")
# axs[0].axhline(-limite, linestyle="--")
# axs[1].axhline(limite, linestyle="--")
# axs[1].axhline(-limite, linestyle="--")
# axs[-1].set_xlabel('Time step')
# plt.tight_layout()
# plt.savefig("examples/tau1_tau2.png")
# plt.show()

# plot_torques_with_bounds()

""" save_trajectory(x_multibody, u_multibody, com_measured, force_FL, force_FR, force_RL, force_RR, solve_time,
                FL_measured, FR_measured, RL_measured, RR_measured,
                FL_references, FR_references, RL_references, RR_references, L_measured, "fulldynamics") """

print(sum(solve_time)/len(solve_time))