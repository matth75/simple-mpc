import pinocchio as pin
import example_robot_data as erd
import aligator
import numpy as np
from aligator import manifolds, constraints, dynamics

import time

from pinocchio.visualize import MeshcatVisualizer as meshviz

from precompute_utils import create_contact_phases, plot_torques

robot = erd.load("go2")


robot = erd.load("go2")
rmodel: pin.Model = robot.model
rdata: pin.Data = rmodel.createData()

nq = rmodel.nq
nv = rmodel.nv
nu = nv - 6     # nb d'actionneurs - freeFlyerJoint(6 DoF) = nb actionneurs commandables

q1 = rmodel.referenceConfigurations["standing"]

q2 = q1.copy()
q2[2] = 0.5

# update positions
pin.forwardKinematics(rmodel, rdata, q1)
pin.updateFramePlacements(rmodel, rdata)

# TODO: add collision with the ground !!
FOOT_FRAME_IDS = {
    fname: rmodel.getFrameId(fname) for fname in ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
}
FOOT_JOINT_IDS = {
    fname: rmodel.frames[fid].parentJoint for fname, fid in FOOT_FRAME_IDS.items()
}

print(FOOT_JOINT_IDS)

umax = rmodel.effortLimit[6:]
# for i in range(len(umax)):
#     umax[i] -= 5

umax[2] -= 5
umax[5] -= 5
umax[8] -= 5
umax[11] -= 5
umin = - umax


constraint_models = []
constraint_datas = []
for fname, fid in FOOT_FRAME_IDS.items():
    joint_id = FOOT_JOINT_IDS[fname]
    pl1 = rmodel.frames[fid].placement
    pl2 = rdata.oMf[fid]
    cm = pin.RigidConstraintModel(
        pin.ContactType.CONTACT_3D,
        rmodel,
        joint_id,
        pl1,
        0,
        pl2,
        pin.LOCAL_WORLD_ALIGNED
    )
    cm.corrector.Kp[:] = (0, 0, 100)
    cm.corrector.Kd[:] = (50, 50, 50)
    constraint_models.append(cm)
    constraint_datas.append(cm.createData())

constraint_models[0].name = "FL_foot"
constraint_models[1].name = "FR_foot"
constraint_models[2].name = "RL_foot"
constraint_models[3].name = "RR_foot"

FL_id = rmodel.getFrameId("FL_foot")
FR_id = rmodel.getFrameId("FR_foot")
RL_id = rmodel.getFrameId("RL_foot")
RR_id = rmodel.getFrameId("RR_foot")


# -------------- Simulation -------------- #

dt = 0.01 # timestep = 10 ms

# settings to solve contacts
proxSettings = pin.ProximalSettings(1e-9, 1e-10, 10)

space = manifolds.MultibodyPhaseSpace(rmodel)

space_standing = np.concatenate((q1, np.zeros(nv)))
space_jumping = np.concatenate((q2, np.zeros(nv)))

com0 = pin.centerOfMass(rmodel, rdata, q1)

# weights
w_x = np.array([
    1000,   # base pos
    1000,
    1000,

    1,   # body orientation
    1,
    1,

    100,   # FL leg
    100,
    100,

    100,   # FR leg
    100,
    100,

    100,   #RL leg
    100,
    100,

    100,   #RR leg
    100,
    100,

    10,   # body vel
    10,
    10,
    10,
    10,
    10,

    10,   # FL vel
    10,
    10,

    10,   # FR vel
    10,
    10,

    10,   # RL vel
    10,
    10,

    10,   #RR vel
    10,
    10
])

w_x = np.diag(w_x)
w_u = np.eye(nu)*1e-1   # pas trop surtout

w_com = 100 * np.ones(3)
w_com = np.diag(w_com)

# actuation matrix
S = np.eye(nv, nu, -6)  # cast u cmd to q

# create velocity constraints
v_ref = pin.Motion().Zero()


# frame velocities to create constraints on landing
frame_vel_FL = aligator.FrameVelocityResidual(
    space.ndx, nu, rmodel, v_ref, FL_id, pin.LOCAL_WORLD_ALIGNED
)[:2]
frame_vel_FR = aligator.FrameVelocityResidual(
    space.ndx, nu, rmodel, v_ref, FR_id, pin.LOCAL_WORLD_ALIGNED
)[:2]

frame_vel_RL = aligator.FrameVelocityResidual(
    space.ndx, nu, rmodel, v_ref, RL_id, pin.LOCAL_WORLD_ALIGNED
)[:2]
frame_vel_RR = aligator.FrameVelocityResidual(
    space.ndx, nu, rmodel, v_ref, RR_id, pin.LOCAL_WORLD_ALIGNED
)[:2]

FL_placement = rdata.oMf[FL_id]
FR_placement = rdata.oMf[FR_id]
RL_placement = rdata.oMf[RL_id]
RR_placement = rdata.oMf[RR_id]

frame_fn_FL = aligator.FramePlacementResidual(
    space.ndx, nu, rmodel, FL_placement, FL_id
)[2]

frame_fn_FR = aligator.FramePlacementResidual(
    space.ndx, nu, rmodel, FR_placement, FR_id
)[2]

frame_fn_RL = aligator.FramePlacementResidual(
    space.ndx, nu, rmodel, RL_placement, RL_id
)[2]

frame_fn_RR = aligator.FramePlacementResidual(
    space.ndx, nu, rmodel, RR_placement, RR_id
)[2]


#############################

T_ds = 50
T_ss = 50
T_fd = 50   # horizon utilisé par le MPC, permet d'ajuster la durée de la 1ere phase de contact

#############################


possible_contacts = {"stand":[True, True, True, True],
                     "FL_up":[False, False, True, True],    # Front legs up
                     "air":[False, False, False, False],    # jumping
                     "RL_up":[True, True, False, False],    # Back legs up
                    }


c_phases = ["stand", "air", "stand", "air"] 


timings = [T_fd + T_ds, T_ss, T_ds * 2] 
cycles = 1  # number of repetitions of the sequence

# get the contacts
contact_phases = [possible_contacts[c] for c in create_contact_phases(c_phases, timings, cycles)]

T_mpc = len(contact_phases)    # nombre de résolutions d'ocp  ~ temps de la simu


def create_dynamics(contact_type):
    """ Creates full dynamics model for each stage of the OCP """
    if contact_type == possible_contacts["stand"]:
        ode = dynamics.MultibodyConstraintFwdDynamics(space, S, constraint_models, proxSettings)

    elif contact_type == possible_contacts["air"]:
        ode = dynamics.MultibodyConstraintFwdDynamics(space, S, [], proxSettings)

    else:
        # return error, this should not happen
        raise ValueError ("Invalid contact type for dynamics creation.")

    dyn_model = dynamics.IntegratorEuler(ode, dt)
    return dyn_model


def createStage(contact, i):
    """ Creates each stage of the OCP, i.e. running cost & constraints for each simulation time step"""

    # create all costs
    state_cost = aligator.StateErrorResidual(space, nu, space_standing)
    ctrl_cost = aligator.ControlErrorResidual(space.ndx, nu)

    # create a running cost instance
    rcost = aligator.CostStack(space, nu)   

    # add all costs to the running cost
    rcost.addCost("state_cost", aligator.QuadraticStateCost(state_cost, w_x))
    rcost.addCost("control_cost", aligator.QuadraticControlCost(space, nu, w_u))

    stm = aligator.StageModel(rcost, create_dynamics(contact))

    # add box constraints for forces
    # stm.addConstraint()
    ctrl_fn = aligator.ControlErrorResidual(space.ndx, np.zeros(nu))
    stm.addConstraint(ctrl_fn, constraints.BoxConstraint(umin, umax))

    # if contact == possible_contacts["stand"]:
    #     ctrl_FL = aligator.MultibodyFrictionConeResidual(space.ndx, rmodel, S, constraint_models, proxSettings, "FL_foot", 0.8)
    #     ctrl_FR = aligator.MultibodyFrictionConeResidual(space.ndx, rmodel, S, constraint_models, proxSettings, "FR_foot", 0.8)
    #     ctrl_RL = aligator.MultibodyFrictionConeResidual(space.ndx, rmodel, S, constraint_models, proxSettings, "RL_foot", 0.8)
    #     ctrl_RR = aligator.MultibodyFrictionConeResidual(space.ndx, rmodel, S, constraint_models, proxSettings, "RR_foot", 0.8)
    #     stm.addConstraint(ctrl_FL, constraints.NegativeOrthant())
    #     stm.addConstraint(ctrl_FR, constraints.NegativeOrthant())
    #     stm.addConstraint(ctrl_RL, constraints.NegativeOrthant())
    #     stm.addConstraint(ctrl_RR, constraints.NegativeOrthant())
    # stm.addConstraint(,constraints.)

    # on landing, feet velocities = 0
    if i == T_fd + T_ds + T_ss:
        stm.addConstraint(frame_vel_FL, constraints.EqualityConstraintSet())
        stm.addConstraint(frame_vel_FR, constraints.EqualityConstraintSet())
        stm.addConstraint(frame_vel_RL, constraints.EqualityConstraintSet())
        stm.addConstraint(frame_vel_RR, constraints.EqualityConstraintSet())

        stm.addConstraint(frame_fn_FL, constraints.EqualityConstraintSet())
        stm.addConstraint(frame_fn_FR, constraints.EqualityConstraintSet())
        stm.addConstraint(frame_fn_RL, constraints.EqualityConstraintSet())
        stm.addConstraint(frame_fn_RR, constraints.EqualityConstraintSet())

    return stm


stages = []
for i in range(T_mpc):
    stages.append(createStage(contact_phases[i], i))

term_cost = aligator.CostStack(space, nu)

problem = aligator.TrajOptProblem(space_standing, stages, term_cost)

# terminal constraints
com_pos = aligator.CentroidalCoMResidual(space.ndx, nu, com0)
com_pos = aligator.LinearFunctionComposition(com_pos, np.diag(np.array([0,0,1]))) # just z component
term_stage_cstr = aligator.StageConstraint(com_pos, constraints.EqualityConstraintSet())
problem.addTerminalConstraint(term_stage_cstr)


TOL = 1e-2
mu_init = 1e-8
max_iters = 500
verbose = aligator.VerboseLevel.VERBOSE
solver = aligator.SolverProxDDP(TOL, mu_init,max_iters, verbose=verbose)

# init solver !
solver.setup(problem)

# values of u and x throughout simulation
us_init = [np.zeros(nu) for _ in range(T_mpc)]
xs_init = [space_standing for _ in range(T_mpc + 1)]

# run sim
conv = solver.run(problem, xs_init, us_init)

res = solver.results
print(res)

forces = np.zeros((T_mpc,12))   # 12 linear forces on each leg

for i in range(T_mpc):
    if i not in range(T_fd + T_ds,T_fd + T_ds + T_ss):
        forces[i,:3] = solver.workspace.problem_data.stage_data[i].dynamics_data.continuous_data.constraint_datas[0].contact_force.linear
        forces[i,3:6] = solver.workspace.problem_data.stage_data[i].dynamics_data.continuous_data.constraint_datas[1].contact_force.linear
        forces[i,6:9] = solver.workspace.problem_data.stage_data[i].dynamics_data.continuous_data.constraint_datas[2].contact_force.linear
        forces[i,9:12] = solver.workspace.problem_data.stage_data[i].dynamics_data.continuous_data.constraint_datas[3].contact_force.linear

with open("fd_traj.npy", "wb") as f:
    np.save(f, np.array(res.xs))
    np.save(f, np.array(res.us))
    np.save(f, forces)

# plot_torques(np.array(res.us))

def play_results():
    viz = meshviz(rmodel, robot.collision_model, robot.visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()
    viz.display(q1)
    time.sleep(4)
    qs = [x[:nq] for x in res.xs]
    viz.play(qs, dt)

play_results()
