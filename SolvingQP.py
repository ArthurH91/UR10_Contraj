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


if __name__ == "__main__":

    pin.seed(5)

    # Creation of the robot
    robot_wrapper = RobotWrapper()
    robot, rmodel, gmodel = robot_wrapper(target=True)
    rdata = rmodel.createData()
    gdata = gmodel.createData()

    # Open the viewer
    vis = create_visualizer(robot)

    # Creating the QP 
    T = 6
    QP = QuadratricProblemNLP(robot, rmodel, rdata, gmodel, gdata, T, k1 = 1, k2=100 )

    # Initial configuration
    q0 = pin.randomConfiguration(rmodel)
    robot.q0 = q0
    vis.display(q0)

    # Initial trajectory 
    Q = np.array(q0)
    for i in range(T):
        Q = np.concatenate((Q, q0))

    # Trust region solver
    trust_region_solver = NewtonMethodMt(
        QP.compute_cost, QP.grad, QP.hess, max_iter=100, callback=None)

    res = trust_region_solver(Q)
    list_fval_mt, list_gradfkval_mt, list_alphak_mt, list_reguk = trust_region_solver._fval_history, trust_region_solver._gradfval_history, trust_region_solver._alphak_history, trust_region_solver._reguk_history
    traj = trust_region_solver._xval_k

    # Scipy solver
    mini = fmin(QP.compute_cost, Q, full_output = True)

    # Trajectory of the Marc Toussaint method 

    print("Press enter for displaying the trajectory of the newton's method from Marc Toussaint")
    display_last_traj(traj, rmodel.nq)


    plt.subplot(411)
    plt.plot(list_fval_mt, "-ob", label="Marc Toussaint's method")
    plt.yscale("log")
    plt.ylabel("Cost")
    plt.legend()

    plt.subplot(412)
    plt.plot(list_gradfkval_mt, "-ob", label="Marc Toussaint's method")
    plt.yscale("log")
    plt.ylabel("Gradient")
    plt.legend()

    plt.subplot(413)
    plt.plot(list_alphak_mt,  "-ob", label="Marc Toussaint's method")
    plt.yscale("log")
    plt.ylabel("Alpha")
    plt.legend()

    plt.subplot(414)
    plt.plot(list_reguk, "-ob", label="Marc Toussaint's method")
    plt.yscale("log")
    plt.ylabel("Regularization")
    plt.xlabel("Iterations")
    plt.legend()

    plt.suptitle(
        " Comparison Newton's method and Marc Toussaint's Newton method")
    plt.show()

    print(trust_region_solver._xval_k)