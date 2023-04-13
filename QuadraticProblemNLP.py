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

from RobotWrapper import RobotWrapper
from create_visualizer import create_visualizer

# This class is for defining the optimization problem and computing the cost function, its gradient and hessian.


class QuadratricProblemNLP():
    def __init__(self, robot, rmodel: pin.Model, rdata: pin.Data, gmodel: pin.GeometryModel, gdata: pin.GeometryData, T : int, k1 = 1, k2 = 1):
        """Initialize the class with the models and datas of the robot.

        Parameters
        ----------
        robot : pin.Robot
            Model of the robot, used for robot.q0
        rmodel : pin.Model
            Model of the robot
        rdata : pin.Data
            Data of the model of the robot
        gmodel : pin.GeometryModel
            Geometrical model of the robot
        gdata : pin.GeometryData
            Geometrical data of the model of the robot
        T : int
            Number of steps for the trajectory
        k1 : float
            Factor of penalisation of the principal cost
        k2 : float
            Factor of penalisation of the terminal cost
        """
        self._robot = robot
        self._rmodel = rmodel
        self._rdata = rdata
        self._gmodel = gmodel
        self._gdata = gdata
        self._T = T
        self._k1 = k1
        self._k2 = k2

        # Storing the IDs of the frames of the end effector and the target

        self._TargetID = self._rmodel.getFrameId('target')
        assert (self._TargetID < len(self._rmodel.frames))

        self._EndeffID = self._rmodel.getFrameId('endeff')
        assert (self._EndeffID < len(self._rmodel.frames))

        # Storing the cartesian pose of the target
        self._target = self._rdata.oMf[self._TargetID].translation



    def _distance_endeff_target(self, q: np.ndarray):
        """Compute distance from a configuration q, from the end effector to the target. 
        Here, the distance is calculated by the difference between the cartesian position of the end effector and the target.

        Parameters
        ----------
        q : np.ndarray
            Array of configuration of the robot, size rmodel.nq.

        Returns
        -------
        residual : np.ndarray
            Array of the distance at a configuration q, size 3. 
        """

        # Forward kinematics of the robot at the configuration q.
        pin.framesForwardKinematics(self._rmodel, self._rdata, q)

        # Obtaining the cartesian position of the end effector.
        p = self._rdata.oMf[self._EndeffID].translation
        return p - self._target
    

    
    def compute_residuals(self, Q: np.ndarray):
        """Compute the residuals of the cost function.
        """

        self._Q = Q

        # Computing the initial residual (q0 - q_init)

        self._initial_residual = self._get_q_iter_from_Q(0) - self._robot.q0

        # Penalizing the initial residual
        self._initial_residual *= self._k1

        # Computing the principal residual 
        self._principal_residual = self._get_difference_between_q_iter(0)

        for iter in range(1,self._T):
            self._principal_residual = np.concatenate( (self._principal_residual, self._get_difference_between_q_iter(iter)), axis=None)

        # Computing the terminal residual

        # Obtaining the last q from Q, mandatory to compute the terminal residual
        q_T = self._get_q_iter_from_Q(self._T)

        # Computing the residual 
        self._terminal_residual = ( self._k2 ** 2 / 2 ) * self._distance_endeff_target(q_T)

        # Adding the terminal residual to the whole residual
        self._residual = np.concatenate( (self._initial_residual,self._principal_residual, self._terminal_residual), axis = None)

        return self._residual
    
    def compute_cost(self, Q: np.ndarray):
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

        self._Q = Q
        res = self.compute_residuals(Q)

        self._initial_cost = 0.5 * np.linalg.norm(self._initial_residual)
        self._principal_cost = 0.5 * np.linalg.norm(self._principal_residual) ** 2 
        self._terminal_cost = 0.5 * np.linalg.norm(self._terminal_residual) ** 2
        self._cost = self._terminal_cost + self._principal_cost
        return self._cost


    def _get_q_iter_from_Q(self, iter: int):
        """Returns the iter-th configuration vector q_iter in the Q array.

        Args:
            iter (int): Index of the q_iter desired.

        Returns:
            q_iter (np.ndarray): Array of the configuration of the robot at the iter-th step.
        """
        q_iter = np.array((self._Q[self._rmodel.nq * iter: self._rmodel.nq * (iter+1)]))
        return q_iter
    

    def _get_difference_between_q_iter(self, iter: int):
        """Returns the difference between the q_iter and q_iter+1 in the array self.Q

        Parameters
        ----------
        iter : int
            Index of the q_iter desired.

        """
        return self._get_q_iter_from_Q(iter + 1) - self._get_q_iter_from_Q(iter)
    
    def _compute_derivative_principal_residuals(self):
        """Computes the derivatives of the principal and initial residuals that are in a matrix, as proved easily mathematically, this matrix is made out of :
        - a matrix ((nq.(T+1) +3) x (nq.(T+1))) where the diagonal is filled with 1  
        - a matrix ((nq.(T+1) +3) x (nq.(T+1))) where the diagonal under the diagonal 0 is filled with -1  

        Returns
        -------
        _derivative_principal_residuals : np.ndarray
            matrix describing the principal residuals derivatives
        """
        _derivative_principal_residuals = self._k1 * np.eye(self._rmodel.nq * (self._T+1) + 3, self._rmodel.nq * (self._T +1)) - np.eye(
            self._rmodel.nq * (self._T+1) + 3, self._rmodel.nq * (self._T+1), k=-self._rmodel.nq)
        
        # Replacing the last -1 by 0 because it goes an iteration too far.
        _derivative_principal_residuals[-3:, -6:] = np.zeros((3,6))
        return _derivative_principal_residuals

    def _compute_derivative_terminal_residuals(self):
        """Computes the derivatives of the terminal residuals, which are for now the jacobian matrix from pinocchio.

        Returns
        -------
        self._derivative_terminal_residuals : np.ndarray
            matrix describing the terminal residuals derivativess
        """
        # Getting the q_terminal from Q 
        q_terminal = self._get_q_iter_from_Q(self._T)

        # Computing the joint jacobian from pinocchio, used as the terminal residual derivative
        _derivative_terminal_residuals = self._k2 **2 * pin.computeJointJacobian(self._rmodel, self._rdata, q_terminal, 6)[:3, :]
        return _derivative_terminal_residuals

    def _compute_derivative_residuals(self):
        """Computes the derivatives of the residuals

        Returns
        -------
        derivative_residuals : np.ndarray
            matrix describing the derivative of the residuals
        """

        # Computing the principal residuals
        self._derivative_residuals = self._compute_derivative_principal_residuals()

        # Computing the terminal residuals 
        _derivative_terminal_residuals = self._compute_derivative_terminal_residuals()

        # Modifying the residuals to include the terminal residuals computed before
        self._derivative_residuals[-3:, -6:] = _derivative_terminal_residuals


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
        self._Q = Q
        _ = self.compute_residuals(Q)
        self._compute_derivative_residuals()

        self.gradval = self._derivative_residuals.T @ self._residual
        return self.gradval

    def hess(self, Q : np.ndarray):
        """Returns the hessian of the cost function.
        """
        self._Q = Q
        _ = self.compute_residuals(Q)
        self._compute_derivative_residuals()
        self.hessval = self._derivative_residuals.T @ self._derivative_residuals

        return self.hessval
    

if __name__ == "__main__":

    # Setting up the environnement 
    robot_wrapper = RobotWrapper()
    robot, rmodel, gmodel = robot_wrapper(target=True)
    rdata = rmodel.createData()
    gdata = gmodel.createData()
    vis = create_visualizer(robot)


    q = pin.randomConfiguration(rmodel)

    pin.framesForwardKinematics(rmodel, rdata, q)

    # THIS STEP IS MANDATORY OTHERWISE THE FRAMES AREN'T UPDATED
    pin.updateGeometryPlacements(rmodel, rdata, gmodel, gdata, q)
    vis.display(q)

    q0 = np.array([1, 1, 1, 1, 1, 1])
    q1 = np.array([2.1, 2.1 ,2.1 ,2.1,2.1,2.1])
    q2 = np.array([3.3, 3.3 ,3.3 ,3.3,3.3,3.3])
    q3 = np.array([4,4,4,4,4,4])

    Q = np.concatenate((q0, q1, q2, q3))
    T = int((len(Q) - 1) / rmodel.nq) 

    QP = QuadratricProblemNLP(robot, rmodel, rdata, gmodel, gdata, T=T, k1 = 10, k2=1 )

    QP._Q = Q

