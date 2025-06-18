from simple_mpc import RobotModelHandler, RobotDataHandler
from aligator import dynamics, manifolds, constraints
import aligator
import pinocchio as pin
import numpy as np

# TODO: create a class to handle OCP resolution

class Go2CentroidalOCP:
    def __init__(self, model_handler:RobotModelHandler):
        self.model_handler = model_handler
        self.data_handler = RobotDataHandler(model_handler)
        self.rmodel = model_handler.getModel()
        self.rdata = self.rmodel.createData()
        self.nq = self.rmodel.nq
        self.nv = self.rmodel.nv
        self.qref = model_handler.getReferenceState()[:self.nq]
        self.nu = 12    # 4 3D contact forces
        self.nx = 9

        FL_id = self.rmodel.getFrameId("FL_foot")
        FR_id = self.rmodel.getFrameId("FR_foot")
        RL_id = self.rmodel.getFrameId("RL_foot")
        RR_id = self.rmodel.getFrameId("RR_foot")
        self.feet_names = list(model_handler.getFeetNames())
        self.feet_ids = [FL_id, FR_id, RL_id, RR_id]

        self.mass = pin.computeTotalMass(self.rmodel)
        self.gravity = np.array([0,0, -9.81])
        self.force_size = 3
        self.dt = 0.01

        self.space = manifolds.VectorSpace(self.nx)
        self.com0 = pin.centerOfMass(self.rmodel, self.rdata, self.qref)

        # weights associated to the runnings costs
        w_control = np.array([   # 3D forces *4 legs = 12 = nu
            1,1,1,
            1,1,1,
            1,1,1,
            1,1,1
        ])
        self.w_control = np.diag(w_control) * 0.01
        self.w_com = np.diag([0,0,0])    # no constraint on com right now

        # force robot to be "stable"
        self.w_lin = np.diag(np.array([0.01, 0.01, 1]))  
        self.w_ang = np.diag(np.array([0.01, 0.01, 1]))
        self.w_linear_acc = 0.01 * np.eye(3)
        self.w_angular_acc = np.diag(np.array([0.01, 1, 0.01]))

        self.setupSolver()

    def createDynamics(self, contact_map):
        """ Creates centroidal dynamic model for each stage of the OCP """
        ode = dynamics.CentroidalFwdDynamics(self.space, self.mass, self.gravity, contact_map, self.force_size)   # force size = 3
        dyn_model = dynamics.IntegratorEuler(ode, self.dt)
        return dyn_model


    def createStage(self, contact, feet_pose, ur):
        """ Creates each stage of the OCP, i.e. running cost & constraints for each simulation time step"""
        # defines which contacts are active and where
        contact_pose = [foot.translation for foot in feet_pose]
        contact_map = aligator.ContactMap(self.feet_names, contact, contact_pose)

        # residuals of COM, linear & angular momentum / accelerations
        centroidal_com = aligator.CentroidalCoMResidual(self.nx, self.nu, self.com0)
        linear_mom = aligator.LinearMomentumResidual(self.nx, self.nu, np.zeros(3))
        angular_mom = aligator.AngularMomentumResidual(self.nx, self.nu, np.zeros(3))
        linear_acc = aligator.CentroidalAccelerationResidual(self.nx, self.nu, self.mass, self.gravity, contact_map, self.force_size)
        angular_acc = aligator.AngularAccelerationResidual(self.nx, self.nu, self.mass, self.gravity, contact_map, self.force_size)

        # create a running cost instance
        rcost = aligator.CostStack(self.space, self.nu)   

        # add all costs to the running cost
        rcost.addCost("state_cost", aligator.QuadraticControlCost(self.space, ur, self.w_control))
        rcost.addCost("com_cost", aligator.QuadraticResidualCost(self.space, centroidal_com, self.w_com))
        rcost.addCost("linear_mom_cost", aligator.QuadraticResidualCost(self.space, linear_mom, self.w_lin))
        rcost.addCost("angular_mom_cost", aligator.QuadraticResidualCost(self.space, angular_mom, self.w_ang))
        rcost.addCost("angular_acc_cost", aligator.QuadraticResidualCost(self.space, angular_acc, self.w_angular_acc))
        rcost.addCost("linear_acc_cost", aligator.QuadraticResidualCost(self.space, linear_acc, self.w_linear_acc))

        # create the stage model associated to the running cost and dynamics
        stm = aligator.StageModel(rcost, self.createDynamics(contact_map))

        # no specific constraints (contact constraints already included in contact_map)

        return stm
    
    def setupProblem(self, x0, contact_phases):
        feet_pose = [self.rdata.oMf[idx].copy() for idx in self.feet_ids]
        T_mpc = len(contact_phases)
        uref = np.zeros(self.nu)   # only zeros should work with simple-mpc
        # create simulation stages (for each contact phase)
        stages = []
        for i in range(T_mpc):
            stages.append(self.createStage(contact_phases[i], feet_pose, uref))
        # add an empty terminal cost
        term_cost = aligator.CostStack(self.space, self.nu)

        # create an OCP instance
        self.problem = aligator.TrajOptProblem(x0, stages, term_cost)

        """ Add some terminal constraints to guarantee the stability of the robot at the end of the horizon"""

        # vitesse et moment angulaire = 0 au bout de l'horizon
        linear_mom = aligator.LinearMomentumResidual(self.nx, self.nu, np.zeros(3))
        term_stage_cstr = aligator.StageConstraint(linear_mom, constraints.EqualityConstraintSet())
        self.problem.addTerminalConstraint(term_stage_cstr)

        angular_mom = aligator.AngularMomentumResidual(self.nx, self.nu, np.zeros(3))
        term_stage_cstr = aligator.StageConstraint(angular_mom, constraints.EqualityConstraintSet())
        self.problem.addTerminalConstraint(term_stage_cstr)

        # angular momentum acceleration = 0
        contact_pose = [foot.translation for foot in feet_pose]
        contact_map = aligator.ContactMap(self.feet_names, contact_phases[-1], contact_pose)

        angular_acc = aligator.AngularAccelerationResidual(self.nx, self.nu, self.mass, self.gravity, contact_map, self.force_size)
        term_stage_cstr = aligator.StageConstraint(angular_acc, constraints.EqualityConstraintSet())
        self.problem.addTerminalConstraint(term_stage_cstr)

    def setupSolver(self):
        TOL = 1e-5
        mu_init = 1e-8 

        max_iters = 50  # easy move, should not take too many iterations
        verbose = aligator.VerboseLevel.VERBOSE
        self.solver = aligator.SolverProxDDP(TOL, mu_init, verbose=verbose)
        #solver = aligator.SolverFDDP(TOL, verbose=verbose)
        self.solver.rollout_type = aligator.ROLLOUT_LINEAR
        #print("LDLT algo choice:", solver.ldlt_algo_choice)
        self.solver.linear_solver_choice = aligator.LQ_SOLVER_PARALLEL #LQ_SOLVER_SERIAL
        self.solver.force_initial_condition = True
        self.solver.setNumThreads(2)
        self.solver.max_iters = max_iters
    
    def runOCP(self, x0, contact_phases):
        self.setupProblem(x0, contact_phases)
        
        T_mpc = len(contact_phases)
        uref = np.zeros(self.nu)   # only zeros should work with simple-mpc

        self.solver.setup(self.problem)
        # warm start
        us_init = [uref for _ in range(T_mpc)]
        xs_init = [x0] * (T_mpc + 1)

        self.solver.run(
            self.problem,
            xs_init,
            us_init,
        )

        res = self.solver.results
        return res
    
    def getCentroidalState(self, q):
        pin.forwardDynamics(self.rmodel, self.rdata, q)
        pin.updateFramePlacements(self.rmodel, self.rdata)
        x = self.data_handler.getCentroidalState()
        return x
    

""" testing """

def testing():
    import example_robot_data as erd
    from go2_utils import create_contact_phases

    robot_wrapper = erd.load("go2")
    model_handler = RobotModelHandler(robot_wrapper.model, "standing", "root_joint")
    data_handler = RobotDataHandler(model_handler)

    x_centr = np.array([-1.29876364e-02,  2.03856673e-04,  3.25087124e-01, -1.35146731e-01,
        2.81344715e-02,  1.76018137e+01, -2.60906889e-04, -1.30279521e-01,
        1.73656433e-03])
    
    g = Go2CentroidalOCP(model_handler)

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

    T_ds, T_ss = 50, 30

    timings = [int(T_ds/2), T_ss, int(T_ds/2)]
    cycles = 2  # number of repetitions of the sequence

    # get the contacts
    contact_phases = [possible_contacts[c] for c in create_contact_phases(c_phases, timings, cycles)]
    contact_phasesOCP = [list(possible_contacts[c].values()) for c in create_contact_phases(c_phases, timings, cycles)]

    print(contact_phasesOCP[30:])


if __name__ == "__main__":
    testing()