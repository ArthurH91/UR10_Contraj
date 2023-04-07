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
    def __init__(self, rmodel: pin.Model, rdata: pin.Data, gmodel: pin.GeometryModel, gdata: pin.GeometryData, T : float, k1 = 1, k2 = 1):
        """Initialize the class with the models and datas of the robot.

        Parameters
        ----------
        rmodel : pin.Model
            Model of the robot
        rdata : pin.Data
            Data of the model of the robot
        gmodel : pin.GeometryModel
            Geometrical model of the robot
        gdata : pin.GeometryData
            Geometrical data of the model of the robot
        T : float
            Number of steps for 
        k1 : float
            Factor of penalisation of the principal cost
        k2 : float
            Factor of penalisation of the terminal cost
        """
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
        """Returns the residuals of the cost function.

        Parameters
        ----------
        Q : np.ndarray
            Array of shape (T*robot.nq) in which all the configurations of the robot are, in a single column.
        
        Returns:
        -------
        residuals (np.ndarray): Array of size (T*robot.nq + 3) defined by the difference between the t-th configuration and the t+1-th configuration and by the terminal residual, which is the distance betwee, the end effector at the end of the optimization iteration and the target. 
        """
        self._Q = Q

        # Computing the principal residual 
        
        # Initializing the _residual array.
        self._residual = self._get_difference_between_q_iter(0)

        for iter in range(1,self._T - 1):
            self._residual = np.concatenate( (self._residual, self._get_difference_between_q_iter(iter)), axis=None)

        # Penalizing the principal residual 
        self._residual *= self._k1

        # Saving the principal residual 
        self._principal_residual = self._residual.copy()

        # Computing the terminal residual

        # Obtaining the last q from Q, mandatory to compute the terminal residual
        q_T = self._get_q_iter_from_Q(self._T-1)

        # Computing the residual 
        self._terminal_residual = ( self._k2 ** 2 / 2 ) * self._distance_endeff_target(q_T)

        # Adding the terminal residual to the whole residual
        self._residual = np.concatenate( (self._residual, self._terminal_residual), axis = None)

        return self._residual
    
    def compute_cost(self):
        """Computes the cost of the QP.

        Parameters
        ----------
        Q : np.ndarray
            Array of shape (T*robot.nq) in which all the configurations of the robot are, in a single column.

        Returns
        -------
        self._cost : float
            Sum of the costs 
        """

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
    


if __name__ == "__main__":


    robot_wrapper = RobotWrapper()
    robot, rmodel, gmodel = robot_wrapper(target=True)
    rdata = rmodel.createData()
    gdata = gmodel.createData()
    vis = create_visualizer(robot)

    q = pin.randomConfiguration(rmodel)
    q1 = pin.randomConfiguration(rmodel)
    q2 = pin.randomConfiguration(rmodel)

    pin.framesForwardKinematics(rmodel, rdata, q)

    # THIS STEP IS MANDATORY OTHERWISE THE FRAMES AREN'T UPDATED
    pin.updateGeometryPlacements(rmodel, rdata, gmodel, gdata, q)
    vis.display(q)

    q1 = np.array([0, 0 ,0 ,0,0,0])
    q2 = np.array([1, 1 ,1 ,1,1,1])
    q3 = np.array([3, 3 ,3 ,3,3,3])

    Q = np.concatenate((q1, q2, q3))

    QP = QuadratricProblemNLP(rmodel, rdata, gmodel, gdata, vis, T=3, k1 = 10, k2=100 )
    QP._Q = Q