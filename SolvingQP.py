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
import pinocchio as pin
import time
from scipy.optimize import fmin
import matplotlib.pyplot as plt
import pinocchio.casadi as cpin
import casadi

from RobotWrapper import RobotWrapper
from create_visualizer import create_visualizer
from QuadraticProblemNLP import QuadratricProblemNLP
from NewtonMethodMarcToussaint import NewtonMethodMt
from Solver import Solver


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



def display_last_traj(Q: np.ndarray, nq : int):
    """Display the trajectory computed by the solver

    Parameters
    ----------
    Q : np.ndarray
        Optimization vector.
    nq : int
        size of q_iter
    """
    for iter in range(int(len(Q)/nq)):
        q_iter = get_q_iter_from_Q(Q,iter,nq)
        vis.display(q_iter)
        input()

def solve_with_casadi(QP):
    # HYPERPARAMS
    T = QP._T
    k1 = QP._k1
    k2 = QP._k2

    #def solve_with_casadi(QP):
    ### CASADI HELPERS
    cmodel = cpin.Model(QP._rmodel)
    cdata = cmodel.createData()
    cq = casadi.SX.sym('q',QP._rmodel.nq)

    cpin.framesForwardKinematics(cmodel, cdata,cq)
    endeff = casadi.Function('p',[cq],[cdata.oMf[QP._EndeffID].translation])


    ### CASADI PROBLEM

    opti = casadi.Opti()
    # Decision variables
    qs = [ opti.variable(QP._rmodel.nq) for model in range(T+1) ]

    residuals = \
        [ (k1**2/2)*(qs[0]-q0) ] \
        + [ (k1**2/2)*(qa-qb) for (qa,qb) in zip(qs[1:],qs[:-1]) ] \
        + [ (k2**2/2)*(endeff(qs[-1])-QP._target) ]
    residuals = casadi.vertcat(*residuals)

    ### Optim
    opti.minimize(casadi.sumsqr(residuals))
    #for x in xs: opti.set_initial(x,x0)

    opti.solver("ipopt") # set numerical backend
    # Caution: in case the solver does not converge, we are picking the candidate values
    # at the last iteration in opti.debug, and they are NO guarantee of what they mean.
    try:
        sol = opti.solve_limited()
        qs_sol = np.concatenate([ opti.value(q) for q in qs ])
        residuals_sol = opti.value(residuals)
    except:
        print('ERROR in convergence, plotting debug info.')
        qs_sol = np.concatenate([ opti.debug.value(q) for q in qs ])
        residuals_sol = opti.debug.value(residuals)

    return qs_sol,residuals_sol

if __name__ == "__main__":

    pin.seed(11)

    # Creation of the robot
    robot_wrapper = RobotWrapper()
    robot, rmodel, gmodel = robot_wrapper(target=True)
    rdata = rmodel.createData()
    gdata = gmodel.createData()

    # Open the viewer
    vis = create_visualizer(robot)

    # Creating the QP 
    T = 6
    QP = QuadratricProblemNLP(robot, rmodel, rdata, gmodel, gdata, T, k1 = .01, k2=1 )

    # Initial configuration
    q0 = pin.randomConfiguration(rmodel)
    q0 = np.array([0, -2.5, 2, -1.2, -1.7, 0])
    robot.q0 = q0
    vis.display(q0)

    # Initial trajectory 
    Q0 = np.array(q0)
    for i in range(T):
        Q0 = np.concatenate((Q0, q0))

    # Trust region solver
    trust_region_solver = NewtonMethodMt(
        QP.compute_cost, QP.grad, QP.hess, max_iter=100, callback=None)

    trust_region_solver(Q0)
    list_fval_mt, list_gradfkval_mt, list_alphak_mt, list_reguk = trust_region_solver._fval_history, trust_region_solver._gradfval_history, trust_region_solver._alphak_history, trust_region_solver._reguk_history
    Q_trs = trust_region_solver._xval_k
    residuals_trs = QP.compute_residuals(Q_trs)
    
    # # Scipy solver
    # mini = fmin(QP.compute_cost, Q, full_output = True)

    # Trust region solver with finite difference
    # trust_region_solver_nd = NewtonMethodMt(
    #     QP.compute_cost, QP._grad_numdiff, QP._hess_numdiff, max_iter=100, callback=None)
    # res = trust_region_solver_nd(Q)
    # list_fval_mt_nd, list_gradfkval_mt_nd, list_alphak_mt_nd, list_reguk_nd = trust_region_solver_nd._fval_history, trust_region_solver_nd._gradfval_history, trust_region_solver_nd._alphak_history, trust_region_solver_nd._reguk_history
    # traj_nd = trust_region_solver_nd._xval_k

    # Trajectory of the Marc Toussaint method 


    Q_casadi,residuals_casadi = solve_with_casadi(QP)
    
    # print("Press enter for displaying the trajectory of the newton's method from Marc Toussaint")
    # display_last_traj(traj, rmodel.nq)

    # print("Now the trajectory of the same method but with the num diff")
    # display_last_traj(traj_nd, rmodel.nq)

    # plt.subplot(411)
    # plt.plot(list_fval_mt, "-ob", label="Marc Toussaint's method")
    # plt.plot(list_fval_mt_nd, "-or", label="Finite difference method")
    # plt.yscale("log")
    # plt.ylabel("Cost")
    # plt.legend()

    # plt.subplot(412)
    # plt.plot(list_gradfkval_mt, "-ob", label="Marc Toussaint's method")
    # plt.plot(list_gradfkval_mt_nd, "-or", label="Finite difference method")
    # plt.yscale("log")
    # plt.ylabel("Gradient")
    # plt.legend()

    # plt.subplot(413)
    # plt.plot(list_alphak_mt,  "-ob", label="Marc Toussaint's method")
    # plt.plot(list_alphak_mt_nd,  "-or", label="Finite difference method")
    # plt.yscale("log")
    # plt.ylabel("Alpha")
    # plt.legend()

    # plt.subplot(414)
    # plt.plot(list_reguk, "-ob", label="Marc Toussaint's method")
    # plt.plot(list_reguk_nd, "-or", label="Finite difference method")
    # plt.yscale("log")
    # plt.ylabel("Regularization")
    # plt.xlabel("Iterations")
    # plt.legend()

    # plt.suptitle(
    #     " Comparison between Marc Toussaint's Newton method and finite difference method")
    # plt.show()

    # # print(trust_region_solver._xval_k)
