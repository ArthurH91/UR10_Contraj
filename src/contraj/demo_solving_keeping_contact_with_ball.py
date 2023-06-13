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
from scipy.optimize import fmin, fmin_bfgs
import matplotlib.pyplot as plt
import hppfcl

from wrapper_robot import RobotWrapper
from wrapper_meshcat import MeshcatWrapper
from quadratic_problem_nlp_keeping_contact_with_ball import (
    QuadratricProblemKeepingTContactWithBall,
)
from solver_newton_mt import SolverNewtonMt
from solver_casadi import CasadiSolver
from utils import display_last_traj, numdiff

# * ### HYPERPARMS
T = 6
WEIGHT_Q0 = 0.001
WEIGHT_QINF = 0.01
WEIGHT_DQ = 0.001
WEIGHT_TERM_POS = 6

EPS = 1e-7
MAX_ITER = 50

WITH_DISPLAY = True
WITH_PLOT = True
WITH_POS_CHECK = False  # Check the initial position and the final position of the robot
WITH_NUMDIFF = True


### * HELPERS (Finite difference comutation of the gradient and the hessian)


def grad_numdiff(Q: np.ndarray):
    return numdiff(QP.cost, Q)


def hess_numdiff(Q: np.ndarray):
    return numdiff(grad_numdiff, Q)


if __name__ == "__main__":
    # Creation of the robot
    robot_wrapper = RobotWrapper()
    robot, rmodel, gmodel = robot_wrapper()
    rdata = rmodel.createData()
    gdata = gmodel.createData()

    # Initial configuration, target and final position
    INITIAL_CONFIG = np.array(
        [4.86472901, -4.18478519, -0.87082197, -6.65534205, 4.77033766, 0.64460685]
    )
    TARGET = pin.SE3(
        np.array(
            [
                [-0.36817572, 0.11708538, 0.92235441, 0.13780169],
                [0.3358343, 0.94180947, 0.01449976, 0.2982887],
                [-0.86698441, 0.31509671, -0.38607266, -0.99107095],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    )
    FINAL_CONFIG = np.array(
        [
            -1632.7728959,
            2903.63614872,
            -5044.37929551,
            -4532.82071308,
            -3960.11656974,
            -4083.44749639,
        ]
    )

    # Shaping the target with HPPFCL
    TARGET_SHAPE = hppfcl.Sphere(5e-2)

    # Obtaining the SE3 pose of the end effector at the final configuration
    pin.forwardKinematics(rmodel, rdata, FINAL_CONFIG)
    pin.updateFramePlacements(rmodel, rdata)

    END_POS = rdata.oMf[rmodel.getFrameId("endeff")]

    # Creating the QP
    QP = QuadratricProblemKeepingTContactWithBall(
        robot,
        rmodel,
        gmodel,
        q0=INITIAL_CONFIG,
        end_pos=END_POS,
        target=TARGET,
        target_shape=TARGET_SHAPE,
        T=T,
        weight_q0=WEIGHT_Q0,
        weight_endeffpos=WEIGHT_QINF,
        weight_dq=WEIGHT_DQ,
        weight_term_pos=WEIGHT_TERM_POS,
    )

    # Generating the meshcat visualizer
    MeshcatVis = MeshcatWrapper()
    vis = MeshcatVis.visualize(TARGET, robot=robot)

    if WITH_POS_CHECK:
        # Displaying the initial configuration of the robot
        input("Press enter to display the initial configuration")
        vis.display(INITIAL_CONFIG)
        input("Press enter to display the end configuration")
        vis.display(FINAL_CONFIG)
        input("Press enter to start the optimization")

    # Displaying the initial configuration of the robot
    vis.display(INITIAL_CONFIG)

    # Initial trajectory
    Q0 = np.concatenate([INITIAL_CONFIG] * (T + 1))

    # Trust region solver

    trust_region_solver = SolverNewtonMt(
    QP.cost,
    QP.grad,
    QP.hess,
    max_iter=MAX_ITER,
    callback=None,
    verbose=True,
    eps=EPS,
    )
    res = trust_region_solver(Q0)
    list_fval_mt, list_gradfkval_mt, list_alphak_mt, list_reguk = (
        trust_region_solver._fval_history,
        trust_region_solver._gradfval_history,
        trust_region_solver._alphak_history,
        trust_region_solver._reguk_history,
    )
    Q = trust_region_solver._xval_k

    if WITH_NUMDIFF:
        # Trust region solver with finite difference
        trust_region_solver_nd = SolverNewtonMt(
            QP.cost,
            QP.grad_numdiff,
            QP.hess_numdiff,
            max_iter=MAX_ITER,
            callback=None,
            verbose=True,
            eps=EPS,
        )
        res = trust_region_solver_nd(Q0)
        list_fval_mt_nd, list_gradfkval_mt_nd, list_alphak_mt_nd, list_reguk_nd = (
            trust_region_solver_nd._fval_history,
            trust_region_solver_nd._gradfval_history,
            trust_region_solver_nd._alphak_history,
            trust_region_solver_nd._reguk_history,
        )
        Q_nd = trust_region_solver_nd._xval_k

    if WITH_DISPLAY:
        print("Trajectory with the TRS method")
        display_last_traj(vis, Q, INITIAL_CONFIG, T)
        if WITH_NUMDIFF:
            print("Now the trajectory of the same method but with the num diff")
            display_last_traj(vis, Q_nd, INITIAL_CONFIG, T)

    if WITH_PLOT:
        plt.subplot(411)
        plt.plot(list_fval_mt, "-ob", label="TRS method")
        plt.plot(list_fval_mt_nd, "-or", label="Finite difference method")
        plt.yscale("log")
        plt.ylabel("Cost")
        plt.legend()

        plt.subplot(412)
        plt.plot(list_fval_mt, "-ob", label="TRS method")
        plt.plot(list_gradfkval_mt_nd, "-or", label="Finite difference method")
        plt.yscale("log")
        plt.ylabel("Gradient")
        plt.legend()

        plt.subplot(413)
        plt.plot(list_fval_mt, "-ob", label="TRS method")
        plt.plot(list_alphak_mt_nd, "-or", label="Finite difference method")
        plt.yscale("log")
        plt.ylabel("Alpha")
        plt.legend()

        plt.subplot(414)
        plt.plot(list_fval_mt, "-ob", label="TRS method")
        plt.plot(list_reguk_nd, "-or", label="Finite difference method")
        plt.yscale("log")
        plt.ylabel("Regularization")
        plt.xlabel("Iterations")
        plt.legend()

        plt.suptitle(
            " Comparison between Marc Toussaint's Newton method and finite difference method"
        )
        plt.show()
