""" Script to generate centroidal ocp and execute mpc scheme"""

import numpy as np
import aligator
import pinocchio as pin

from simple_mpc import RobotModelHandler

import os
import sys
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

space = manifolds.MultibodyPhaseSpace(rmodel)     # used for init of com

nx = space.nx

x0 = space.neutral()
print(x0)
u0 = np.zeros(nu)   # warm start

pin.forwardKinematics(rmodel, rdata, q0)
pin.updateFramePlacements(rmodel, rdata)
mass = pin.computeTotalMass(rmodel)

# create OCP problem
gravity = np.array([0, 0, -9.81])
mu = 0.8                 # friction coefficient : use it ?


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
T_ss = 50   # for testing purposes
T_ds = 50

# c_phases = ["stand", "FL_up", "air", "RL_up", "stand"]
c_phases = ["stand", "air", "stand"]

# timings = [30, 10, 30, 2, 30]
timings = [T_ds, T_ss, 70]
cycles = 1  # number of repetitions of the sequence

# get the contacts
contact_phases = [possible_contacts[c] for c in go2.create_contact_phases(c_phases, timings, cycles)]

T_mpc = len(contact_phases)    # nombre de résolutions d'ocp  ~ temps de la simu

f0 = -mass*gravity[2] / nk  # force applied on one leg when all legs are touching the ground
f2 = f0 * 2   # force applied when 2 legs are touching the ground

print(contact_phases)

