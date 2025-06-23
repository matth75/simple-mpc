""" Script to generate centroidal ocp and execute mpc scheme"""

import numpy as np
import aligator
import pinocchio as pin

from simple_mpc import RobotModelHandler

import go2_utils as go2

from aligator import (manifolds, 
                    dynamics, 
                    constraints)

import example_robot_data as erd

robot = erd.load("go2")

URDF_FILENAME = "go2.urdf"
URDF_SUBPATH = "/go2_description/urdf/go2.urdf"

# create model and data intances
rmodel = robot.model
rdata = rmodel.createData()

base_joint_name = "root_joint"

FL_id = rmodel.getFrameId("FL_foot")
FR_id = rmodel.getFrameId("FR_foot")
RL_id = rmodel.getFrameId("RL_foot")
RR_id = rmodel.getFrameId("RR_foot")

feet_ids = [FL_id, FR_id, RL_id, RR_id]
feet_name = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

base_id = rmodel.getFrameId("root_joint")
torso_id = rmodel.getFrameId("base")

model_handler = RobotModelHandler(rmodel, "standing", base_joint_name)

q0 = rmodel.referenceConfigurations["standing"]

# torque limits on all joints of the robot
low_limits = rmodel.lowerPositionLimit[7:]
up_limits = rmodel.upperPositionLimit[7:]

nq = rmodel.nq
nv = rmodel.nv
force_size = 3          # 3D contact forces (feets)
nk = 4                  # 4 legs, 4 contact forces
nu = nk * force_size    # dimension of base

nx = 9          # dimension 9 for centroidal dynamics (see paper)

space = manifolds.VectorSpace(nx)
space_multibody = manifolds.MultibodyPhaseSpace(rmodel)     # used for init of com

x0 = space.neutral()
print(x0)
u0 = np.zeros(nu)   # warm start

# update com position
pin.forwardKinematics(rmodel, rdata, q0)
pin.updateFramePlacements(rmodel, rdata)
com0 = pin.centerOfMass(rmodel, rdata, q0)
x0[:3] = com0.copy()

# go2.plot_com_3d(np.array(com0), show_plot=True)

# create OCP problem
gravity = np.array([0, 0, -9.81])
mu = 0.8                 # friction coefficient : use it ?

# compute reference (starting ?) forces
mass = pin.computeTotalMass(rmodel)
f_ref = np.array([0, 0, -mass*gravity[2] / nk])

# initial values of u0
for i in range(nk):
    u0[force_size * i + 2] = -gravity[2] * mass / nk

# rmodel.names[1:].tolist() : exclude universe

controlled_joints = ['root_joint', 'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 'FR_hip_joint', 
 'FR_thigh_joint', 'FR_calf_joint', 'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint', 
 'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint']

controlled_ids = [rmodel.getJointId(j) for j in controlled_joints[1:]]    # cant control root joint

simu_step = 1e-3

dt = 0.01   # 10 ms = temps de calcul d'ocp
Nsimu = int(dt/simu_step)   # nombre de simulation entre deux résolutions d'ocp

""" Create contact phases """

# format is CONTACT_TYPE = Front Left, Front Right, Rear Left, Rear Right
possible_contacts = {"stand":[True, True, True, True],
                     "FL_up":[False, False, True, True],    # Front legs up
                     "air":[False, False, False, False],    # jumping
                     "RL_up":[True, True, False, False],    # Back legs up
                    }

# contact phases and corresponding timings
T_ss = 30   # for testing purposes
T_ds = 20

# c_phases = ["stand", "FL_up", "air", "RL_up", "stand"]
c_phases = ["stand", "air", "stand"]

# timings = [30, 10, 30, 2, 30]
timings = [T_ds, T_ss, T_ds]
cycles = 2  # number of repetitions of the sequence

# get the contacts
contact_phases = [possible_contacts[c] for c in go2.create_contact_phases(c_phases, timings, cycles)]

T_mpc = len(contact_phases)    # nombre de résolutions d'ocp  ~ temps de la simu

f0 = -mass*gravity[2] / nk  # force applied on one leg when all legs are touching the ground
f2 = f0 * 2   # force applied when 2 legs are touching the ground

# standard forces (3D) applied to all 4 legs 
possible_u = {"stand":np.array([0,0,f0] * nk),
            "FL_up":np.array([0, 0, f2] *2 + [0, 0, 0]*2),
            "air":np.zeros(nu),
            "RL_up":np.array([0, 0, 0] *2 + [0, 0, f2]*2)
            }

u0 = np.array([0,0,f0, 0,0, f0, 0,0,f0, 0,0,f0])
# pas nécessaire
uref =[]
for c in contact_phases:
    if c == [True, True, True, True]:
        uref.append(u0)
    else:
        uref.append(np.zeros(12))


""" Define feet trajectory """
swing_apex = 0.15
x_forward = 0.0
y_forward = 0.030
foot_yaw = 0
y_gap = 0.18
x_depth = 0

""" Cost weights and stage creation """

# weights associated to the runnings costs
w_control = np.array([   # 3D forces *4 legs = 12 = nu
    1,1,1,
    1,1,1,
    1,1,1,
    1,1,1
])

A_forces = np.array([0,0,1,0,0,1,0,0,1,0,0,1])
A_forces = np.diag(A_forces)

b_forces = np.array([0,0,1,0,0,1,0,0,1,0,0,1])*60
w_control = np.diag(np.array([0.1,0.1,10,
                              0.1,0.1,10,
                              0.1,0.1,10,
                              0.1,0.1,10])) 
# w_control = np.eye(12)*0.1
w_com = np.diag([0,0,1])    # no constraint on com right now

# force robot to be "stable"
w_lin = np.diag(np.array([0.01, 0.01, 1]))  
w_ang = np.diag(np.array([0.01, 0.01, 1]))
w_linear_acc = 0.01 * np.eye(3)
w_angular_acc = np.diag(np.array([0.01, 1, 0.01]))

def create_dynamics(contact_map):
    """ Creates centroidal dynamic model for each stage of the OCP """
    ode = dynamics.CentroidalFwdDynamics(space, mass, gravity, contact_map, force_size)
    dyn_model = dynamics.IntegratorEuler(ode, dt)
    return dyn_model


def createStage(contact, contact1, i, feet_pose, ur):
    """ Creates each stage of the OCP, i.e. running cost & constraints for each simulation time step"""
    # defines which contacts are active and where
    contact_pose = [foot.translation for foot in feet_pose]
    contact_map = aligator.ContactMap(feet_name, contact, contact_pose)

    # residuals of COM, linear & angular momentum / accelerations
    centroidal_com = aligator.CentroidalCoMResidual(nx, nu, np.array(com0))
    forces = aligator.ControlErrorResidual(space.ndx, nu)
    linear_mom = aligator.LinearMomentumResidual(nx, nu, np.zeros(3))
    angular_mom = aligator.AngularMomentumResidual(nx, nu, np.zeros(3))
    linear_acc = aligator.CentroidalAccelerationResidual(nx, nu, mass, gravity, contact_map, force_size)
    angular_acc = aligator.AngularAccelerationResidual(nx, nu, mass, gravity, contact_map, force_size)

    # create a running cost instance
    rcost = aligator.CostStack(space, nu)   

    # add all costs to the running cost
    rcost.addCost("state_cost", aligator.QuadraticControlCost(space, ur, w_control))
    # rcost.addCost("com_cost", aligator.QuadraticResidualCost(space, centroidal_com, w_com))
    if contact == [True, True, True, True]:
        rcost.addCost("linear_mom_cost", aligator.QuadraticResidualCost(space, linear_mom, w_lin))
        rcost.addCost("angular_mom_cost", aligator.QuadraticResidualCost(space, angular_mom, w_ang))
        rcost.addCost("angular_acc_cost", aligator.QuadraticResidualCost(space, angular_acc, w_angular_acc))
        rcost.addCost("linear_acc_cost", aligator.QuadraticResidualCost(space, linear_acc, w_linear_acc))
    

    # create the stage model associated to the running cost and dynamics
    stm = aligator.StageModel(rcost, create_dynamics(contact_map))

    # no specific constraints (contact constraints already included in contact_map)
    # centroidal_com = aligator.LinearFunctionComposition(centroidal_com, -np.diag(np.array([0,0,1]))) # -h(x)
    forces = aligator.LinearFunctionComposition(forces, np.eye(12), -b_forces)
    # stm.addConstraint(centroidal_com, constraints.NegativeOrthant()) # h(x)<0

    if contact == [True, True, True, True] and contact1 == [False, False, False, False]:
        stm.addConstraint(centroidal_com, constraints.NegativeOrthant())
        # stm.addConstraint(forces, constraints.NegativeOrthant())

    # if i == (T_ds + T_ss + 15):
    #     stm.addConstraint(centroidal_com2, constraints.EqualityConstraintSet())
        # stm.addConstraint(forces, constraints.NegativeOrthant())
    return stm

feet_pose = [rdata.oMf[idx].copy() for idx in feet_ids]

t1,t2 = 0, 80
contact_phases  = contact_phases[t1:t2]
uref = uref[t1:t2]
T_mpc = len(contact_phases)    # nombre de résolutions d'ocp  ~ temps de la simu

# create simulation stages (for each contact phase)
stages = []
for i in range(T_mpc):
    stages.append(createStage(contact_phases[i], contact_phases[(i+1)%T_mpc], i, feet_pose, uref[i]))

# add an empty terminal cost
term_cost = aligator.CostStack(space, nu)

# create an OCP instance
T_mpc = len(stages)
# x0[5] = 10
problem = aligator.TrajOptProblem(x0, stages, term_cost)

""" Add some terminal constraints to guarantee the stability of the robot at the end of the horizon"""

if contact_phases[-1] == [True, True, True, True]:
    # vitesse et moment angulaire = 0 au bout de l'horizon
    linear_mom = aligator.LinearMomentumResidual(nx, nu, np.zeros(3))
    term_stage_cstr = aligator.StageConstraint(linear_mom, constraints.EqualityConstraintSet())
    problem.addTerminalConstraint(term_stage_cstr)

    ang_mom = aligator.AngularMomentumResidual(nx, nu, np.zeros(3))
    term_stage_cstr = aligator.StageConstraint(ang_mom, constraints.EqualityConstraintSet())
    problem.addTerminalConstraint(term_stage_cstr)

# position en z du com = z du com de référence
com_pos = aligator.CentroidalCoMResidual(space.ndx, nu, com0)
com_pos = aligator.LinearFunctionComposition(com_pos, np.diag(np.array([0,0,1]))) # just z component
term_stage_cstr = aligator.StageConstraint(com_pos, constraints.EqualityConstraintSet())
problem.addTerminalConstraint(term_stage_cstr)

angular_mom = aligator.AngularMomentumResidual(nx, nu, np.zeros(3))
term_stage_cstr = aligator.StageConstraint(angular_mom, constraints.EqualityConstraintSet())
# problem.addTerminalConstraint(term_stage_cstr)

# # position en z du com = z du com de référence
# com_pos = aligator.StateErrorResidual(space, nu, com0)[2]
# term_stage_cstr = aligator.StageConstraint(com_pos, constraints.EqualityConstraintSet())
# problem.addTerminalConstraint(term_stage_cstr)


# angular momentum acceleration = 0
contact_pose = [foot.translation for foot in feet_pose]
contact_map = aligator.ContactMap(feet_name, contact_phases[-1], contact_pose)

angular_acc = aligator.AngularAccelerationResidual(nx, nu, mass, gravity, contact_map, force_size)
term_stage_cstr = aligator.StageConstraint(angular_acc, constraints.EqualityConstraintSet())
# problem.addTerminalConstraint(term_stage_cstr)

""" Parametrize the solver"""

TOL = 1e-5
mu_init = 1e-8 

max_iters = 50  # easy move, should not take too many iterations
verbose = aligator.VerboseLevel.VERBOSE
solver = aligator.SolverProxDDP(TOL, mu_init, verbose=verbose)
#solver = aligator.SolverFDDP(TOL, verbose=verbose)
solver.rollout_type = aligator.ROLLOUT_LINEAR
#print("LDLT algo choice:", solver.ldlt_algo_choice)
solver.linear_solver_choice = aligator.LQ_SOLVER_PARALLEL #LQ_SOLVER_SERIAL
solver.force_initial_condition = True
solver.setNumThreads(2)
solver.max_iters = max_iters

solver.setup(problem)


# warm start
us_init = [u0 for _ in range(T_mpc)]
xs_init = [x0] * (T_mpc + 1)

solver.run(
    problem,
    xs_init,
    us_init,
)

res = solver.results
print(res)

xs = np.array(res.xs)
us = np.array(res.us)

go2.plot_results(xs)
go2.plot_forces(us)

