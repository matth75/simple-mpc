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
        self.qref = self.rmodel.referenceConfigurations["standing"]
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
        # dont forget to update frame placements, else com0 reference will be 0,0,0 !!!!
        pin.forwardKinematics(self.rmodel, self.rdata, self.qref)
        pin.updateFramePlacements(self.rmodel, self.rdata)
        self.com0 = pin.centerOfMass(self.rmodel, self.rdata, self.qref)
        self.x0 = self.space.neutral()
        self.x0[:3] = self.com0.copy()

        # weights associated to the runnings costs
        w_control = np.array([   # 3D forces *4 legs = 12 = nu
            1,1,1,
            1,1,1,
            1,1,1,
            1,1,1
        ])
        self.w_control = np.diag(np.array([0.1,0.1,10,
                              0.1,0.1,10,
                              0.1,0.1,10,
                              0.1,0.1,10])) 
        self.w_com = np.diag([1,1,100])    # no constraint on com right now

        # force robot to be "stable"
        self.w_lin = np.diag(np.array([0.1, 0.1, 1]))  
        self.w_ang = np.diag(np.array([0.1, 0.1, 1]))
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
        centroidal_com = aligator.CentroidalCoMResidual(self.nx, self.nu,np.array(self.com0) + np.array([0,0,0.01]))
        linear_mom = aligator.LinearMomentumResidual(self.nx, self.nu, np.zeros(3))
        angular_mom = aligator.AngularMomentumResidual(self.nx, self.nu, np.zeros(3))
        linear_acc = aligator.CentroidalAccelerationResidual(self.nx, self.nu, self.mass, self.gravity, contact_map, self.force_size)
        angular_acc = aligator.AngularAccelerationResidual(self.nx, self.nu, self.mass, self.gravity, contact_map, self.force_size)

        # create a running cost instance
        rcost = aligator.CostStack(self.space, self.nu)   

        # add all costs to the running cost
        rcost.addCost("control_cost", aligator.QuadraticControlCost(self.space, ur, self.w_control))
        # rcost.addCost("com_cost", aligator.QuadraticResidualCost(self.space, centroidal_com, self.w_com))
        if contact == [True, True, True, True]:
            # rcost.addCost("centr_cost", aligator.QuadraticResidualCost(self.space, centroidal_com, self.w_com))
            rcost.addCost("linear_mom_cost", aligator.QuadraticResidualCost(self.space, linear_mom, self.w_lin))
            rcost.addCost("angular_mom_cost", aligator.QuadraticResidualCost(self.space, angular_mom, self.w_ang))
            rcost.addCost("angular_acc_cost", aligator.QuadraticResidualCost(self.space, angular_acc, self.w_angular_acc))
            rcost.addCost("linear_acc_cost", aligator.QuadraticResidualCost(self.space, linear_acc, self.w_linear_acc))

        # create the stage model associated to the running cost and dynamics
        stm = aligator.StageModel(rcost, self.createDynamics(contact_map))

        # if contact == [True, True, True, True]:
        #     stm.addConstraint(centroidal_com, constraints.NegativeOrthant())

        # no specific constraints (contact constraints already included in contact_map)

        return stm
    
    def setupProblem(self, x0, contact_phases):
        feet_pose = [self.rdata.oMf[idx].copy() for idx in self.feet_ids]
        T_mpc = len(contact_phases)
        f0 = -self.mass*self.gravity[2] / 4  # force applied on one leg when all legs are touching the ground
        u0 = np.array([0,0,f0, 0,0, f0, 0,0,f0, 0,0,f0])
        # pas nécessaire
        uref =[]
        for c in contact_phases:
            if c == [True, True, True, True]:
                uref.append(u0)
            else:
                uref.append(np.zeros(12))
        # create simulation stages (for each contact phase)
        stages = []
        for i in range(T_mpc):
            stages.append(self.createStage(contact_phases[i], feet_pose, uref[i]))

        # add an empty terminal cost
        term_cost = aligator.CostStack(self.space, self.nu)

        # create an OCP instance
        self.problem = aligator.TrajOptProblem(x0, stages, term_cost)

        """ Add some terminal constraints to guarantee the stability of the robot at the end of the horizon"""

        if contact_phases[-1] == [True, True, True, True]:
            # vitesse et moment angulaire = 0 au bout de l'horizon
            linear_mom = aligator.LinearMomentumResidual(self.nx, self.nu, np.zeros(3))
            term_stage_cstr = aligator.StageConstraint(linear_mom, constraints.EqualityConstraintSet())
            self.problem.addTerminalConstraint(term_stage_cstr)

        # position en z du com = z du com de référence
        com_pos = aligator.CentroidalCoMResidual(self.space.ndx, self.nu, self.com0)
        com_pos = aligator.LinearFunctionComposition(com_pos, np.diag(np.array([0,0,1]))) # -h(x)
        term_stage_cstr = aligator.StageConstraint(com_pos, constraints.EqualityConstraintSet())
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
        uref = np.zeros(self.nu)   # warm start only zeros should work with simple-mpc 
        # uref[2] = -self.gravity[2]*self.mass/4
        # uref[5] = -self.gravity[2]*self.mass/4
        # uref[8] = -self.gravity[2]*self.mass/4
        # uref[11] = -self.gravity[2]*self.mass/4


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
    from go2_utils import create_contact_phases, plot_results, plot_forces

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

    c_phases = ["stand", "stand"]

    T_ds, T_ss = 50, 30

    timings = [int(T_ds/2), T_ss, int(T_ds/2)]
    cycles = 5  # number of repetitions of the sequence

    # get the contacts
    contact_phases = [possible_contacts[c] for c in create_contact_phases(c_phases, timings, cycles)]
    contact_phasesOCP = [list(possible_contacts[c].values()) for c in create_contact_phases(c_phases, timings, cycles)]

    space = manifolds.VectorSpace(g.nx)
    space_multibody = manifolds.MultibodyPhaseSpace(g.rmodel)     # used for init of com

    x0 = space.neutral()
    print(x0)
    u0 = np.zeros(g.nu)   # warm start

    # update com position
    pin.forwardKinematics(g.rmodel, g.rdata, g.qref)
    pin.updateFramePlacements(g.rmodel, g.rdata)
    com0 = pin.centerOfMass(g.rmodel, g.rdata, g.qref)
    x0[:3] = com0.copy()
    print(x0)

    res = g.runOCP(g.x0, contact_phasesOCP)

    plot_forces(np.array(res.us))
    plot_results(np.array(res.xs))

if __name__ == "__main__":
    testing()