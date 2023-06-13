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
import copy

import hppfcl
import pydiffcol

from wrapper_robot import RobotWrapper
from utils import get_q_iter_from_Q, get_difference_between_q_iter, numdiff

# This class is for defining the optimization problem and computing the cost function, its gradient and hessian of the following QP problem :
# Touching the ball while going from an initial position to another


class QuadratricProblemKeepingTContactWithBall:
    def __init__(
        self,
        robot,
        rmodel: pin.Model,
        gmodel: pin.GeometryModel,
        q0: np.array,
        end_pos: np.array,
        target: pin.SE3,
        target_shape: hppfcl.ShapeBase,
        T: int,
        weight_q0: float,
        weight_endeffpos: float,
        weight_dq: float,
        weight_term_pos: float,
    ):
        """Initialize the class with the models and datas of the robot.

        Parameters
        ----------
        robot : pin.Robot
            Model of the robot, used for robot.q0
        rmodel : pin.Model
            Model of the robot
        q0: np.array
            Initial configuration of the robot
        target: np.array
            Target position for the end effector
        target_shape: hppfcl.ShapeBase
            hppfcl.ShapeBase of the target
        T : int
            Number of steps for the trajectory
        weight_q0 : float
            Factor of penalisation of the initial cost (q_0 - q0)**2
        weight_dq : float
            Factor of penalisation of the running cost (q_t+1 - q_t)**2
        weight_term_pos : float
            Factor of penalisation of the terminal cost (p(q_T) - target)**
        """
        self._robot = robot
        self._rmodel = rmodel
        self._gmodel = gmodel
        self._rdata = rmodel.createData()
        self._gdata = gmodel.createData()

        self._q0 = q0
        self._end_pos = end_pos
        self._T = T
        self._target = target
        self._target_shape = target_shape
        self._weight_q0 = weight_q0
        self._weight_endeffpos = weight_endeffpos
        self._weight_dq = weight_dq
        self._weight_term_pos = weight_term_pos

        # Storing the IDs of the frame of the end effector

        self._EndeffID = self._rmodel.getFrameId("endeff")
        self._EndeffID_geom = self._gmodel.getGeometryId("endeff_geom")
        assert self._EndeffID_geom < len(self._gmodel.geometryObjects)
        assert self._EndeffID < len(self._rmodel.frames)

        self._target_pos = self._target.translation

    def cost(self, Q: np.ndarray):
        """Computes the cost of the QP.

        Parameters
        ----------
        Q : np.ndarray
            Array of shape (T*rmodel.nq) in which all the configurations of the robot are, in a single column.

        Returns
        -------
        self._cost : float
            Sum of the costs
        """

        ###* INITIAL RESIDUAL
        # Computing the distance between q0 and q_init to make sure the robot 
        # starts at the right place
        self._initial_residual = get_q_iter_from_Q(Q, 0, self._rmodel.nq) - self._q0

        # Penalizing the initial residual
        self._initial_residual *= self._weight_q0

        ###* RUNNING RESIDUAL
        # Running residuals are computed by diffenciating between q_th and q_th +1
        self._principal_residual = (
            get_difference_between_q_iter(Q, 0, self._rmodel.nq) * self._weight_dq
        )
        for iter in range(1, self._T):
            self._principal_residual = np.concatenate(
                (
                    self._principal_residual,
                    get_difference_between_q_iter(Q, iter, self._rmodel.nq)
                    * self._weight_dq,
                ),
                axis=None,
            )

        ###* TOUCHING BALL RESIDUALS
        # Computing the distance between the configurations and the target,
        # in order to stay close to the ball

        # Distance request for pydiffcol
        self._req = pydiffcol.DistanceRequest()
        self._res = pydiffcol.DistanceResult()

        self._req.derivative_type = pydiffcol.DerivativeType.FirstOrderRS

        # Creating a list of the terminal residuals :
        self._ball_touching_residual = np.zeros((3 * self._rmodel.nq))

        for k in range(1,self._T-1):
            # Obtaining the k configuration of Q
            q_k = get_q_iter_from_Q(Q, k, self._rmodel.nq)

            # Forward kinematics of the robot at the configuration q_k.
            pin.framesForwardKinematics(self._rmodel, self._rdata, q_k)
            pin.updateGeometryPlacements(
                self._rmodel, self._rdata, self._gmodel, self._gdata, q_k
            )

            # Obtaining the cartesian position of the end effector.
            self.endeff_Transform = self._rdata.oMf[self._EndeffID]
            self.endeff_Shape = self._gmodel.geometryObjects[
                self._EndeffID_geom
            ].geometry

            self._req = pydiffcol.DistanceRequest()
            self._res = pydiffcol.DistanceResult()
            #
            dist_endeff_target = pydiffcol.distance(
                self.endeff_Shape,
                self.endeff_Transform,
                self._target_shape,
                self._target,
                self._req,
                self._res,
            )

            self._ball_touching_residual[k * 3 : (k + 1) * 3] = (
                self._weight_term_pos
            ) * self._res.w


        ###* FINAL POSITION RESIDUAL
        # Computing the distance between the final pose and the pose of the end effector 
        # at the end of the trajectory.

        q_final = (
            get_q_iter_from_Q(Q, self._T, self._rmodel.nq)
        )            
        pin.framesForwardKinematics(self._rmodel, self._rdata, q_final)
        pin.updateGeometryPlacements(
                self._rmodel, self._rdata, self._gmodel, self._gdata, q_final
            )
        

        self._final_position_residual = self._rdata.oMf[self._EndeffID].translation - self._end_pos.translation
        self._final_position_residual *= self._weight_endeffpos

        ###* TOTAL RESIDUAL
        self._residual = np.concatenate(
            (
                self._initial_residual,
                self._principal_residual,
                self._ball_touching_residual,
                self._final_position_residual,
            ),
            axis=None,
        )

        ###* COMPUTING COSTS
        self._initial_cost = 0.5 * sum(self._initial_residual**2)
        self._final_cost = 0.5 * sum(self._final_position_residual**2)
        self._principal_cost = 0.5 * sum(self._principal_residual**2)
        self._ball_touching_cost = 0.5 * sum(self._ball_touching_residual**2)
        self.costval = (
            self._initial_cost
            + self._final_cost
            + self._ball_touching_cost
            + self._principal_cost
        )

        return self.costval

    def grad(self, Q: np.ndarray):
        """Returns the grad of the cost function.

        Parameters
        ----------
        Q : np.ndarray
            Array of shape (T*rmodel.nq) in which all the configurations of the robot are, in a single column.

        Returns
        -------
        gradient : np.ndarray
            Array of shape (T*rmodel.nq + 3) in which the values of the gradient of the cost function are computed.
        """

        ### COST AND RESIDUALS
        self.cost(Q)
        nq, T = self._rmodel.nq, self._T

        ### DERIVATIVES OF THE RESIDUALS

        # Computing the derivative of the initial residuals
        self._derivative_initial_residual = np.diag([self._weight_q0] * nq)

        # Computing the derivative of the principal residual
        J_principal = np.zeros((T * nq, (T + 1) * nq))
        np.fill_diagonal(J_principal, -self._weight_dq)
        np.fill_diagonal(J_principal[:, nq:], self._weight_dq)

        self._derivative_principal_residual = J_principal

        self._derivative_touching_residual = np.zeros(((T+1)*3, (T+1)*nq ))
        # Computing the derivative of the ball touching residual
        for i in range(1,T):
            q_i = get_q_iter_from_Q(Q, i, nq)

            # Computing the jacobians in pinocchio
            pin.computeJointJacobians(self._rmodel, self._rdata, q_i)

            # Computing the derivatives of the distance
            _ = pydiffcol.distance_derivatives(
                self.endeff_Shape,
                self.endeff_Transform,
                self._target_shape,
                self._target,
                self._req,
                self._res,
            )

            # Getting the frame jacobian from the end effector in the LOCAL reference frame
            jacobian = pin.computeFrameJacobian(
                self._rmodel, self._rdata, q_i, self._EndeffID, pin.LOCAL
            )

            # The jacobian here is the multiplication of the jacobian of the end effector 
            # and the jacobian of the distance between the end effector and the target
            J = jacobian.T @ self._res.dw_dq1.T
            self._derivative_terminal_residual = self._weight_term_pos * J.T

            self._derivative_touching_residual[3*i:3*(i+1),:]
        
        self._derivative_final_residual = pin.getFrameJacobian(
            self._rmodel, self._rdata, self._EndeffID, pin.LOCAL_WORLD_ALIGNED)[:3]
        

        # Putting them all together

        self._derivative_initial_principal_residual = np.zeros(
            [(T + 1) * nq , (T+1) * nq]
        )

        # Computing the initial residuals
        self._derivative_initial_principal_residual[
            : nq, : nq
        ] = self._derivative_initial_residual

        # Computing the principal residuals
        self._derivative_initial_principal_residual[
            nq:
        ] = self._derivative_principal_residual

        self._derivative_residual =  np.zeros(
            [(T + 1) * (nq + 3), (T+1) * nq])
        
        self._derivative_residual[:(T+1)*nq,:] = self._derivative_initial_principal_residual

        self._derivative_residual[(T+1)*nq:(T+1)*(3+nq)+1,:] = self._derivative_touching_residual
        self._derivative_residual[-3:, -self._rmodel.nq:] =  self._derivative_final_residual


        self.gradval = self._derivative_residual.T @ self._residual

        gradval_numdiff = self.grad_numdiff(Q) 
        print(f"grad val : {np.linalg.norm(self.gradval)} \n grad val numdiff : {np.linalg.norm(gradval_numdiff)}")
        assert np.linalg.norm(self.gradval - gradval_numdiff, np.inf) < 1e-5
        return self.gradval

    def hess(self, Q: np.ndarray):
        """Returns the hessian of the cost function with regards to the gauss newton approximation"""

        self.cost(Q)
        self.grad(Q)
        self.hessval = self._derivative_residual.T @ self._derivative_residual

        return self.hessval

    def grad_numdiff(self, Q: np.ndarray):
        return numdiff(self.cost, Q)

    def hess_numdiff(self, Q: np.ndarray):
        return numdiff(self.grad_numdiff, Q)


if __name__ == "__main__":
    from utils import numdiff

    # Setting up the environnement
    robot_wrapper = RobotWrapper()
    robot, rmodel, gmodel = robot_wrapper()
    rdata = rmodel.createData()
    gdata = gmodel.createData()

    q = pin.randomConfiguration(rmodel)

    pin.framesForwardKinematics(rmodel, rdata, q)

    # THIS STEP IS MANDATORY OTHERWISE THE FRAMES AREN'T UPDATED
    pin.updateGeometryPlacements(rmodel, rdata, gmodel, gdata, q)

    q0 = np.array([1, 1, 1, 1, 1, 1])
    q1 = np.array([2.1, 2.1, 2.1, 2.1, 2.1, 2.1])
    q2 = np.array([3.3, 3.3, 3.3, 3.3, 3.3, 3.3])
    q3 = np.array([4, 4, 4, 4, 4, 4])

    Q = np.concatenate((q0, q1, q2, q3))
    T = int((len(Q) - 1) / rmodel.nq)
    p = pin.SE3.Random()

    # The target shape is a ball of 5e-2 radii at the TARGET position
    TARGET_SHAPE = hppfcl.Sphere(5e-2)

    QP = QuadratricProblemKeepingTContactWithBall(
        robot,
        rmodel,
        gmodel,
        q0=q,
        qinf=q3,
        target=p,
        target_shape=TARGET_SHAPE,
        T=T,
        weight_q0=5,
        weight_qinf=0.1,
        weight_dq=0.1,
        weight_term_pos=10,
    )

    QP._Q = Q

    def grad_numdiff(Q: np.ndarray):
        return numdiff(QP.cost, Q)

    def hess_numdiff(Q: np.ndarray):
        return numdiff(grad_numdiff, Q)

    cost = QP.cost(Q)
    grad = QP.grad(Q)
    gradval_numdiff = grad_numdiff(Q)
    hessval_numdiff = hess_numdiff(Q)
    print(np.linalg.norm(grad - gradval_numdiff, np.inf))
    assert np.linalg.norm(grad - gradval_numdiff, np.inf) < 1e-4
