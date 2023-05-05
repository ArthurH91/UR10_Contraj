# 2-Clause BSD License

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
from numpy.linalg import norm
import pinocchio as pin
import time
from scipy.optimize import fmin,fmin_bfgs
import matplotlib.pyplot as plt

from robot_wrapper import RobotWrapper
from meshcat_wrapper import MeshcatWrapper
from problem_traj import QuadratricProblemNLP
from solver_newton_mt import SolverNewtonMt
from dedicated_solver_casadi import CasadiSolver
from generate_reachable_target import generateReachableTarget

# ### HYPERPARMS
T = 6
WEIGHT_Q0 =  .001
WEIGHT_DQ = .001
WEIGHT_TERM_POS = 4

TARGET = np.array([ -.155,-.815,.456 ])
TARGET = "random"

INITIAL_CONFIG = np.array([0, -2.5, 2, -1.2, -1.7, 0])
INITIAL_CONFIG = "random"

SEED = abs(int(np.sin(time.time() % 6.28) * 1000))
SEED = 573 # TRS does not perfectly converge, slight difference with IpOpt
SEED = 1 # Perfect convergence to solution, immediate convergence of IpOpt (with WS)
print(f'SEED = {SEED}' )

WITH_DISPLAY = False
WITH_PLOT = False
WITH_NUMDIFF_SOLVE = False
WARMSTART_IPOPT_WITH_TRS = False

# ### HELPERS
def get_q_iter_from_Q(Q : np.ndarray, iter: int, nq: int):
    """Returns the iter-th configuration vector q_iter in the Q array.

        Args:
            Q (np.ndarray): Optimization vector.
            iter (int): Index of the q_iter desired.
            nq (int): size of q_iter

        Returns:
            q_iter (np.ndarray): Array of the configuration of the robot at the iter-th step.
        """
    q_iter = np.array((Q[nq * iter: nq * (iter+1)]))
    return q_iter

def display_last_traj(Q: np.ndarray, T : int, dt = None):
    """Display the trajectory computed by the solver

    Parameters
    ----------
    Q : np.ndarray
        Optimization vector.
    nq : int
        size of q_iter
    """
    for q_iter in [robot.q0]+np.split(Q,T+1):
        vis.display(q_iter)
        if dt is None:
            input()
        else:
            time.sleep(dt)

        
if __name__ == "__main__":

    pin.seed(SEED)

    # Creation of the robot
    robot_wrapper = RobotWrapper()
    robot, rmodel, gmodel = robot_wrapper()
    rdata = rmodel.createData()
    gdata = gmodel.createData()

    # Initial configuration and target
    if INITIAL_CONFIG == "random":
        INITIAL_CONFIG = pin.randomConfiguration(rmodel)
    if TARGET == "random":
        TARGET = generateReachableTarget(rmodel,rdata).translation

        
    # Creating the QP 
    QP = QuadratricProblemNLP(robot, rmodel,
                              q0 = INITIAL_CONFIG,
                              target = TARGET,
                              T = T,
                              weight_q0 = WEIGHT_Q0,
                              weight_dq = WEIGHT_DQ,
                              weight_term_pos = WEIGHT_TERM_POS)
    
    # Generating the meshcat visualizer
    MeshcatVis = MeshcatWrapper()
    vis = MeshcatVis.visualize(p, robot=robot)

    # Displaying the initial configuration of the robot
    vis.display(INITIAL_CONFIG)

    # Initial trajectory 
    Q0 = np.concatenate([ INITIAL_CONFIG ]*(T+1) )

    # Trust region solver
    trust_region_solver = SolverNewtonMt(
        QP.compute_cost, QP.grad, QP.hess, max_iter=100, callback=None)

    
    trust_region_solver(Q0)
    list_fval_mt, list_gradfkval_mt, list_alphak_mt, list_reguk = trust_region_solver._fval_history, trust_region_solver._gradfval_history, trust_region_solver._alphak_history, trust_region_solver._reguk_history
    Q_trs = trust_region_solver._xval_k
    residuals_trs = QP.compute_residuals(Q_trs)

    if WITH_NUMDIFF_SOLVE:
        # # Scipy solver
        mini = fmin_bfgs(QP.compute_cost, Q0, full_output = True)
        Q_fmin = mini

        # Trust region solver with finite difference
        trust_region_solver_nd = SolverNewtonMt(
            QP.compute_cost, QP._grad_numdiff, QP._hess_numdiff, max_iter=100, callback=None)
        res = trust_region_solver_nd(Q0)
        list_fval_mt_nd, list_gradfkval_mt_nd, list_alphak_mt_nd, list_reguk_nd = trust_region_solver_nd._fval_history, trust_region_solver_nd._gradfval_history, trust_region_solver_nd._alphak_history, trust_region_solver_nd._reguk_history
        Q_nd = trust_region_solver_nd._xval_k

    # Casadi+IpOpt solver
    casadiSolver = CasadiSolver(QP)
    Q_casadi,residuals_casadi =  casadiSolver.solve(Q_trs if WARMSTART_IPOPT_WITH_TRS else None)
    J_casadi = casadiSolver.evalJacobian(Q_casadi)

    # ### NUMDIFF unittest
    Qr=np.random.rand((T+1)*rmodel.nq)*6-3
    gnd = QP._grad_numdiff(Qr)
    Jr=casadiSolver.evalJacobian(Qr)
    rr=casadiSolver.evalResiduals(Qr)
    gcas = Jr.T@rr
    galg = QP.grad(Qr)
    assert( norm(rr-QP._residual,np.inf)<1e-9 )
    assert( norm(gcas-galg,np.inf)<1e-9 )
    assert( norm(gnd-galg,np.inf)<1e-3 )


    if WITH_DISPLAY:
        print("Press enter for displaying the trajectory of the newton's method from Marc Toussaint")
        display_last_traj(Q_trs, T)
        if WITH_NUMDIFF_SOLVE:
            print("Now the trajectory of the same method but with the num diff")
            display_last_traj(Q_nd, T)

    if WITH_PLOT:
        plt.subplot(411)
        plt.plot(list_fval_mt, "-ob", label="Marc Toussaint's method")
        if WITH_NUMDIFF_SOLVE:
            plt.plot(list_fval_mt_nd, "-or", label="Finite difference method")
        plt.yscale("log")
        plt.ylabel("Cost")
        plt.legend()
        
        plt.subplot(412)
        plt.plot(list_gradfkval_mt, "-ob", label="Marc Toussaint's method")
        if WITH_NUMDIFF_SOLVE:
            plt.plot(list_gradfkval_mt_nd, "-or", label="Finite difference method")
        plt.yscale("log")
        plt.ylabel("Gradient")
        plt.legend()
        
        plt.subplot(413)
        plt.plot(list_alphak_mt,  "-ob", label="Marc Toussaint's method")
        if WITH_NUMDIFF_SOLVE:
            plt.plot(list_alphak_mt_nd,  "-or", label="Finite difference method")
        plt.yscale("log")
        plt.ylabel("Alpha")
        plt.legend()

        plt.subplot(414)
        plt.plot(list_reguk, "-ob", label="Marc Toussaint's method")
        if WITH_NUMDIFF_SOLVE:
            plt.plot(list_reguk_nd, "-or", label="Finite difference method")
        plt.yscale("log")
        plt.ylabel("Regularization")
        plt.xlabel("Iterations")
        plt.legend()

        plt.suptitle(
            " Comparison between Marc Toussaint's Newton method and finite difference method")
        plt.show()

    np.set_printoptions(precision=3, linewidth=600, suppress=True) 
    print('Optimal trajectory: \n\t', '\n\t '.join([repr(q) for q in np.split(Q_casadi,T+1)]))
    pin.framesForwardKinematics(rmodel,rdata,Q_casadi[-rmodel.nq:])
    print('Terminal position:', rdata.oMf[QP._EndeffID].translation, ' vs ', QP._target)
    print(f'Distance between IpOpt and TRS solvers {norm(Q_casadi-Q_trs,np.inf)} ' )
    if WITH_NUMDIFF_SOLVE:
        print(f'Distance between IpOpt and ND solvers {norm(Q_casadi-Q_nd,np.inf)} ' )
        print(f'Distance between ND and TRS solvers {norm(Q_nd-Q_trs,np.inf)} ' )

    assert( np.allclose(Q_casadi,Q_trs,atol=1e-7,rtol=1e-3) )
    if WITH_NUMDIFF_SOLVE:
        assert( np.allclose(Q_casadi,Q_nd,atol=1e-3,rtol=10) )
        assert( np.allclose(Q_trs,Q_nd,atol=1e-3,rtol=10) )