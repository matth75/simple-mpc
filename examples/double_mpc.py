import aligator.constraints
import numpy as np
from bullet_robot import BulletRobot
from simple_mpc import (
    RobotModelHandler,
    RobotDataHandler,
    FullDynamicsOCP,
    CentroidalOCP,
    MPC,
    IDSolver,
    Interpolator,
    FrictionCompensation
)

import aligator
import example_robot_data as erd
import pinocchio as pin
import time
from utils import extract_forces
import copy

from ocp_centroidal.go2_utils import (create_contact_phases,
                                        plot_results,
                                        compare_predictions, 
                                        compare_forces,
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

# robot data
nq = model_handler.getModel().nq
nv = model_handler.getModel().nv
nu = nv - 6
force_size = 3  # 3D contacts
nk = len(model_handler.getFeetNames())
nf = force_size 

# reference forces on each leg
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

dt = 0.01   # simulation timestep

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

""" Initialise the two OCP problems """

problem_conf_fd = dict(
    timestep=dt,
    w_x=np.diag(w_x),
    w_u=np.eye(u0.size) * 1e-4,
    w_cent=np.diag(np.concatenate((w_cent_lin, w_cent_ang))),
    gravity=gravity,
    force_size=3,
    w_forces=np.diag(w_forces_lin),
    w_frame=w_frame,
    w_com = w_com_fd,    # for CoM tracking !! that's great
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

# length of the horizon (in simulation steps) for FD OCP
T_fd = 50

fd_problem = FullDynamicsOCP(problem_conf_fd, model_handler)
fd_problem.createProblem(model_handler.getReferenceState(), T_fd, force_size, gravity[2], False)

w_control = np.array([   # 3D forces *4 legs = 12 = nu
    1,1,1,
    1,1,1,
    1,1,1,
    1,1,1
])
w_control = np.diag(w_control) * 0.01
w_com_centr = np.diag([0,0,1])    # no constraint on com right now
w_lin = np.diag(np.array([0.01, 0.01, 1]))  
w_ang = np.diag(np.array([0.01, 0.01, 1]))
w_linear_acc = 0.01 * np.eye(3)
w_angular_acc = np.diag(np.array([0.01, 1, 0.01]))


problem_conf_ctr = dict(
    timestep=0.01,
    w_u=w_control,
    w_com=w_com_centr,
    w_linear_mom=w_lin,
    w_angular_mom=w_ang,
    w_linear_acc=w_linear_acc,
    w_angular_acc=w_angular_acc,
    gravity=gravity,
    mu=0.8,
    Lfoot=0.01,
    Wfoot=0.01,
    force_size=force_size,
)

# horizon length for centroidal dynamics OCP
T_ctr = 100
ctr_problem = CentroidalOCP(problem_conf_ctr, model_handler)
ctr_problem.createProblem(data_handler.getCentroidalState(), T_ctr, force_size, gravity[2], True)

# linear_mom = aligator.LinearMomentumResidual(9, 12, np.zeros(3))
# term_stage_cstr = aligator.StageConstraint(linear_mom, aligator.constraints.EqualityConstraintSet())
# ctr_problem.getProblem().addTerminalConstraint(term_stage)

# useless ??
T_ds = 20
T_ss = 80

mpc_conf = dict(
    support_force=-model_handler.getMass() * gravity[2],
    TOL=1e-4,
    mu_init=1e-8,
    max_iters=1,
    num_threads=1,
    swing_apex=0.15,
    T_fly=T_ss,
    T_contact=T_ds,
    timestep=problem_conf_ctr["timestep"],
)

mpc_conf_ctr = dict(
    support_force=-model_handler.getMass() * gravity[2],
    TOL=1e-5,
    mu_init=1e-8,
    max_iters=5,    # interesting
    num_threads=1,
    swing_apex=0.15,
    T_fly=T_ss,
    T_contact=T_ds,
    timestep=problem_conf_ctr["timestep"],
)

mpc_centr = MPC(mpc_conf_ctr, ctr_problem)
mpc_fd = MPC(mpc_conf, fd_problem)


# choose the phases of the motion
c_phases = ["stand", "air", "stand"]

timings = [T_fd, 40, 50]
cycles = 1  # number of repetitions of the sequence

# get the contacts
contact_phases = [possible_contacts[c] for c in create_contact_phases(c_phases, timings, cycles)]
# contact_phasesOCP = [list(possible_contacts[c].values()) for c in create_contact_phases(c_phases, timings, cycles)]



mpc_centr.generateCycleHorizon(contact_phases)
mpc_fd.generateCycleHorizon(contact_phases)

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

device.changeCamera(1.0, 60, -15, [0.6, -0.5, 0.5])


""" Interpolation """
interpolator = Interpolator(model_handler.getModel())

q_meas, v_meas = device.measureState()
x_measured = np.concatenate([q_meas, v_meas])
mpc_fd.getDataHandler().updateInternalData(x_measured, False)
mpc_centr.getDataHandler().updateInternalData(x_measured, False)
x_centr = mpc_fd.getDataHandler().getCentroidalState()  


ref_foot_pose = [mpc_fd.getDataHandler().getRefFootPose(mpc_fd.getModelHandler().getFeetNames()[i]) for i in range(4)]
for pose in ref_foot_pose:
    pose.translation[2] = 0
device.showQuadrupedFeet(*ref_foot_pose)

# burn through the first static states. 
# These states are there to guarantee that the cycling horizon is bigger than the main horizon (T_ctr)
# This is done by creating T_ctr states at the beginning that are standing states.

for t in range(T_fd):
    mpc_centr.iterate(x_measured)

# real simulation begins here
# /!\ T_ctr > T_fd => different loop to burn through the first states of full dynamics needed /!\

res_c = np.array(mpc_centr.xs)

res_fd = []
for s in range(T_fd):
    x_fd = mpc_fd.xs[s]
    data_handler.updateInternalData(x_fd, False)
    res_fd.append(data_handler.getCentroidalState())

res_fd = np.array(res_fd)

# plot_results(res)
# compare_predictions(res_c, res_fd)
# plot_results(res_c)
# plot_forces(np.array(mpc_centr.us))

ee_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
f_refs = []
f_dict = {"FL_foot":None, "FR_foot":None, "RL_foot":None, "RR_foot":None}

""" Storing sim data """
force_FL = []
force_FR = []
force_RL = []
force_RR = []

# v = np.zeros(6)
# v[0] = 0.1
# mpc_centr.velocity_base = v
# mpc_fd.velocity_base = v


comp_times = [ 90, 120]

nsteps = 500    # length of simulation
N_simu = 10     # nb of simulation steps between two OCP solves

i = 0

if True:
    for t in range(nsteps):
        print("Time " + str(t))

        device.moveQuadrupedFeet(
            mpc_fd.getReferencePose(0, "FL_foot").translation,
            mpc_fd.getReferencePose(0, "FR_foot").translation,
            mpc_fd.getReferencePose(0, "RL_foot").translation,
            mpc_fd.getReferencePose(0, "RR_foot").translation,
        )


        # start = time.time()
        f_refs = []
        mpc_centr.iterate(x_measured)
        results = np.array(mpc_centr.xs)
        forces = np.array(mpc_centr.us)
        # need a mise en forme des forces 
        for i,f in enumerate(forces[:50]): 
            # f = [int(_) for _ in f]
            f0 = {"FL_foot":f[:3], "FR_foot":f[3:6], "RL_foot":f[6:9], "RR_foot":f[9:]}
            mpc_fd.forcesReferences[i] = f0

        

        com_predicted = results[:50,:3]
        mom_predicted = results[:50,3:]

        mpc_fd.MomReferences = list(mom_predicted)
        mpc_fd.setComReferences = list(com_predicted)
        
        mpc_fd.iterate(x_measured)

        # end = time.time()
        # solve_time.append(end - start)

        a0 = mpc_fd.getStateDerivative(0)[nv:]
        a1 = mpc_fd.getStateDerivative(1)[nv:]

        forces_vec0 = mpc_fd.getContactForces(0)
        forces_vec1 = mpc_fd.getContactForces(1)
        contact_states = mpc_fd.ocp_handler.getContactState(0)

        force_FL.append(forces_vec0[:3])
        force_FR.append(forces_vec0[3:6])
        force_RL.append(forces_vec0[6:9])
        force_RR.append(forces_vec0[9:12])

        forces = [forces_vec0, forces_vec1]
        ddqs = [a0, a1]
        xss = [mpc_fd.xs[0], mpc_fd.xs[1]]
        uss = [mpc_fd.us[0], mpc_fd.us[1]]


        if t in comp_times: # beginning of the jump : first [False, False, False ,False]
            contact_states = mpc_fd.ocp_handler.getContactState(0)
            # x = mpc_fd.xs[0]
            print(mpc_centr.solver.results)
            # data_handler.updateInternalData(x, False)
            # x_centr = data_handler.getCentroidalState()

            traj_fd = []
            forces_fd = []
            for s in range(T_fd):  # len(mpc.xs) = 51, T = 50
                x_fd = mpc_fd.xs[s]
                data_handler.updateInternalData(x_fd, False)
                traj_fd.append(data_handler.getCentroidalState())
                forces_fd.append(mpc_fd.getContactForces(s))

            traj_fd = np.array(traj_fd)

            compare_predictions(np.array(mpc_centr.xs), traj_fd)
            compare_forces(np.array(mpc_centr.us), np.array(forces_fd))
            # plot_forces(np.array(mpc_centr.us))
            # plot_forces(np.array(forces_fd))


        for j in range(N_simu):
            # time.sleep(0.01)
            delay = j / float(N_simu) * dt

            x_interp = interpolator.interpolateState(delay, dt, xss)
            u_interp = interpolator.interpolateLinear(delay, dt, uss)
            # acc_interp = interpolator.interpolateLinear(delay, dt, ddqs)
            # force_interp = interpolator.interpolateLinear(delay, dt, forces)

            q_meas, v_meas = device.measureState()
            x_measured = np.concatenate([q_meas, v_meas])

            mpc_fd.getDataHandler().updateInternalData(x_measured, True)

            x_centr = mpc_fd.getDataHandler().getCentroidalState()
            mpc_centr.getDataHandler().updateInternalData(x_measured, True) # always x_measured for updateInternalData()

            current_torque = u_interp - 1. * mpc_fd.Ks[0] @ model_handler.difference(
                x_measured, x_interp
            )

            # No QP solve
            device.execute(current_torque)





