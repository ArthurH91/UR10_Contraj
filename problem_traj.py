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

from robot_wrapper import RobotWrapper
from utils import get_q_iter_from_Q, get_difference_between_q_iter

# This class is for defining the optimization problem and computing the cost function, its gradient and hessian.


class QuadratricProblemNLP():
    def __init__(self, robot, rmodel: pin.Model,
                 q0: np.array,
                 target: np.array,
                 T : int,
                 weight_q0: float,
                 weight_dq: float,
                 weight_term_pos: float):
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
        self._rdata = rmodel.createData()

        self._q0 = q0
        self._T = T
        self._target = target
        self._weight_q0 = weight_q0
        self._weight_dq = weight_dq
        self._weight_term_pos = weight_term_pos
        
        # Storing the IDs of the frame of the end effector

        self._EndeffID = self._rmodel.getFrameId('endeff')
        assert (self._EndeffID < len(self._rmodel.frames))


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

        self._Q = Q

        ### INITIAL RESIDUAL 
        ### Computing the distance between q0 and q_init to make sure the robot starts at the right place
        self._initial_residual = get_q_iter_from_Q(self._Q, 0, self._rmodel.nq) - self._q0

        # Penalizing the initial residual
        self._initial_residual *= self._weight_q0


        ### RUNNING RESIDUAL 
        ### Running residuals are computed by diffenciating between q_th and q_th +1
        self._principal_residual = get_difference_between_q_iter(Q, 0, self._rmodel.nq) * self._weight_dq
        for iter in range(1,self._T):
            self._principal_residual = np.concatenate((self._principal_residual, get_difference_between_q_iter(Q, iter, self._rmodel.nq) * self._weight_dq),axis=None)


        ### TERMINAL RESIDUAL 
        ### Computing the distance between the last configuration and the target 

        # Obtaining the last configuration of Q
        q_last = get_q_iter_from_Q(self._Q, self._T, self._rmodel.nq)
        
        # Forward kinematics of the robot at the configuration q.
        pin.framesForwardKinematics(self._rmodel, self._rdata, q_last)

        # Obtaining the cartesian position of the end effector.
        p_endeff = self._rdata.oMf[self._EndeffID].translation

        # Comuting the distance between the target and the end effector
        dist_endeff_target = p_endeff - self._target

        self._terminal_residual = ( self._weight_term_pos ) * dist_endeff_target

        ### TOTAL RESIDUAL
        self._residual = np.concatenate( (self._initial_residual,self._principal_residual, self._terminal_residual), axis = None)

        ### COMPUTING COSTS 
        self._initial_cost = 0.5 * sum(self._initial_residual ** 2)
        self._principal_cost = 0.5 * sum(self._principal_residual ** 2)
        self._terminal_cost = 0.5 * sum(self._terminal_residual ** 2)
        self._cost = self._initial_cost + self._terminal_cost + self._principal_cost

        return self._cost

    
    def _compute_derivative_initial_residuals(self):
        """
        Computes the derivatives of the initial residuals that are in a matrix, 
        which is Eye*weight_q0

        Returns
        -------
        J = Eye*weight_q0
        """

        return np.diag([self._weight_q0]*self._rmodel.nq)
    
    def _compute_derivative_principal_residuals(self):
        """
        Computes the derivatives of the principal  residuals that are in a matrix, 
        as proved easily mathematically, this matrix is made out of :
        - a matrix ((nq.(T+1) +3) x (nq.(T+1))) where the diagonal is filled with 1  
        - a matrix ((nq.(T+1) +3) x (nq.(T+1))) where the diagonal under the diagonal 0 is filled with -1  

        Returns
        -------
        _derivative_principal_residuals : np.ndarray
            matrix describing the principal residuals derivatives
        """

        # Initially, the matrix was created from 2 np.eye (inducing 3 memory allocations
        # _derivative_principal_residuals = (np.eye(self._rmodel.nq * (self._T+1) + 3, self._rmodel.nq * (self._T +1)) - np.eye(
        #     self._rmodel.nq * (self._T+1) + 3, self._rmodel.nq * (self._T+1), k=-self._rmodel.nq))*(self._weight_dq)
        nq,T = self._rmodel.nq,self._T
        J = np.zeros((T*nq,(T+1)*nq))
        np.fill_diagonal(J,-self._weight_dq)
        np.fill_diagonal(J[:,nq:],self._weight_dq)

        return J

    def _compute_derivative_terminal_residuals(self):
        """Computes the derivatives of the terminal residuals, which are for now the jacobian matrix from pinocchio.

        Returns
        -------
        self._derivative_terminal_residuals : np.ndarray
            matrix describing the terminal residuals derivativess
        """
        # Getting the q_terminal from Q 
        q_terminal = get_q_iter_from_Q(self._Q,self._T, self._rmodel.nq)

        # Computing the joint jacobian from pinocchio, used as the terminal residual derivative
        ##_derivative_terminal_residuals = self._weight_term_pos  * pin.computeJointJacobian(self._rmodel, self._rdata, q_terminal, 6)[:3, :]
        pin.computeJointJacobians(self._rmodel, self._rdata, q_terminal)
        J = pin.getFrameJacobian(self._rmodel, self._rdata, self._EndeffID, pin.LOCAL_WORLD_ALIGNED)
        _derivative_terminal_residuals = self._weight_term_pos  * J[:3]

        return _derivative_terminal_residuals

    def _compute_derivative_residuals(self):
        """Computes the derivatives of the residuals

        Returns
        -------
        derivative_residuals : np.ndarray
            matrix describing the derivative of the residuals
        """

        T,nq = self._T,self._rmodel.nq
        
        self._derivative_residuals = np.zeros([ (T+1)*nq+3,(T+1)*nq ])
        
        # Computing the initial residuals
        self._derivative_residuals[:nq,:nq] = self._compute_derivative_initial_residuals()

        # Computing the principal residuals
        self._derivative_residuals[nq:-3,:] = self._compute_derivative_principal_residuals()

        # Computing the terminal residuals 
        self._derivative_residuals[-3:,-nq:] = self._compute_derivative_terminal_residuals()

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
        self.cost(self._Q)
        self._compute_derivative_residuals()

        self.gradval = self._derivative_residuals.T @ self._residual
        return self.gradval

    def hess(self, Q : np.ndarray):
        """Returns the hessian of the cost function.
        """
        self._Q = Q
        self.cost(self._Q)
        self._compute_derivative_residuals()
        self.hessval = self._derivative_residuals.T @ self._derivative_residuals

        return self.hessval
    
    def _numdiff(self, f, x, eps=1e-8):
        """Estimate df/dx at x with finite diff of step eps

        Parameters
        ----------
        f : function handle
            Function evaluated for the finite differente of its gradient.
        x : np.ndarray
            Array at which the finite difference is calculated
        eps : float, optional
            Finite difference step, by default 1e-6

        Returns
        -------
        jacobian : np.ndarray
            Finite difference of the function f at x.
        """
        xc = np.copy(x)
        f0 = np.copy(f(x))
        res = []
        for i in range(len(x)):
            xc[i] += eps
            res.append(copy.copy(f(xc)-f0)/eps)
            xc[i] = x[i]
        return np.array(res).T
    

    def _grad_numdiff(self, Q:np.ndarray):
        """Computing the gradient of the cost with the finite difference method

        Parameters
        ----------
        Q : np.ndarray
            Array of shape (T*rmodel.nq) in which all the configurations of the robot are, in a single column.

        Returns
        -------
        _num_diff_hessval : np.ndarray
            Gradient computed with finite difference method
        """

        self._derivative_residuals_num_diff = self._numdiff(self.compute_residuals, Q)

        self._num_diff_gradval = self._numdiff(self.compute_cost, Q)

        return self._num_diff_gradval
    
    def _hess_numdiff(self, Q: np.ndarray):
        """Computing the hessian value of the cost with the finite difference method

        Parameters
        ----------
        Q : np.ndarray
            Array of shape (T*rmodel.nq) in which all the configurations of the robot are, in a single column.


        Returns
        -------
        _num_diff_hessval : np.ndarray
            Hessian computed with finite difference method
        """

        self._num_diff_hessval = self._numdiff(self._grad_numdiff, Q)
        return self._num_diff_hessval
    

if __name__ == "__main__":

    # Setting up the environnement 
    robot_wrapper = RobotWrapper()
    robot, rmodel, gmodel = robot_wrapper(target=True)
    rdata = rmodel.createData()
    gdata = gmodel.createData()

    q = pin.randomConfiguration(rmodel)

    pin.framesForwardKinematics(rmodel, rdata, q)

    # THIS STEP IS MANDATORY OTHERWISE THE FRAMES AREN'T UPDATED
    pin.updateGeometryPlacements(rmodel, rdata, gmodel, gdata, q)

    q0 = np.array([1, 1, 1, 1, 1, 1])
    q1 = np.array([2.1, 2.1 ,2.1 ,2.1,2.1,2.1])
    q2 = np.array([3.3, 3.3 ,3.3 ,3.3,3.3,3.3])
    q3 = np.array([4,4,4,4,4,4])

    Q = np.concatenate((q0, q1, q2, q3))
    T = int((len(Q) - 1) / rmodel.nq) 
    p = np.array([.1,.2,.3])
    
    QP = QuadratricProblemNLP(robot, rmodel,
                              q0 = q,
                              target = p,
                              T=T,
                              weight_q0 = 5,
                              weight_dq = .1,
                              weight_term_pos = 10)

    QP._Q = Q

    cost = QP.compute_cost(Q)
    grad = QP.grad(Q)
    grad_numdiff = QP._grad_numdiff(Q)
    hessval_numdiff = QP._hess_numdiff(Q)

    assert( np.linalg.norm(grad-grad_numdiff,np.inf) < 1e-4 )
