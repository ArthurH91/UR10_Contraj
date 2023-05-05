import unittest

import numpy as np
import pinocchio as pin
import copy

from problem_traj import QuadratricProblemNLP
from wrapper_robot import RobotWrapper

np.set_printoptions(precision=3, linewidth=300, suppress=True, threshold=10000)

class TestQuadraticProblemNLP(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Setup of the environement
        robot_wrapper = RobotWrapper()
        cls._robot, cls._rmodel, cls._gmodel = robot_wrapper(target=True)
        cls._rdata = cls._rmodel.createData()
        cls._gdata = cls._gmodel.createData()
        cls._T = 2
        cls._QP = QuadratricProblemNLP(
            cls._robot, cls._rmodel, cls._rdata, cls._gmodel, cls._gdata, cls._T)

        # Configuration array reaching the target
        cls._q_target = robot_wrapper._q_target
        cls._q_init = pin.randomConfiguration(cls._rmodel)
        cls._q_inter = pin.randomConfiguration(cls._rmodel)
        cls._Q_target = np.concatenate((cls._q_init, cls._q_inter, cls._q_target))
        cls._QP._Q = cls._Q_target

        # Applying the initial configuration to the robot
        cls._robot.q0 = cls._q_init

        # Variables for computing the residuals
        cls._residual = cls._QP.compute_residuals(cls._Q_target)

        # First, computing the initial residual (ie verifying that the robot has the right initial configuration)
        cls._initial_residual_handmade = cls._QP._k1 * \
            (cls._QP._get_q_iter_from_Q(0) - cls._q_init)
        cls._initial_residual = cls._QP._initial_residual

        # Computing the principal residual (ie the speed of the robot)
        cls._principal_residual = cls._QP._principal_residual
        cls._principal_residual_handmade = np.concatenate(
            (cls._q_inter - cls._q_init, cls._q_target - cls._q_inter))

        # Variables for computing the cost
        cls._cost = cls._QP.compute_cost(cls._Q_target)

        # First, computing the initial cost
        cls._initial_cost = cls._QP._initial_cost
        cls._initial_cost_handmade = 0.5 * \
            np.linalg.norm(cls._initial_residual_handmade) ** 2

        # Computing the principal cost
        cls._principal_cost = cls._QP._principal_cost
        cls._principal_cost_handmade = 0.5 * \
            np.linalg.norm(cls._principal_residual_handmade) ** 2

        # Variables for computing the derivatives of the residuals
        cls._QP._compute_derivative_residuals()

        cls._derivative_principal_residuals = cls._QP._compute_derivative_principal_residuals()
        cls._derivative_residuals = cls._QP._derivative_residuals

        # Computing the gradient and the hessian to compare them to the one computed with finite difference method
        cls._gradient = cls._QP.grad(cls._Q_target)
        cls._hessval = cls._QP.hess(cls._Q_target)

        # Gradient and hessian computed with finite difference
        cls._gradient_numdiff = cls._QP._grad_numdiff(cls._Q_target)
        cls._hessval_numdiff = cls._QP._hess_numdiff(cls._Q_target)



    # Tests 

    def test_get_q_iter_from_Q(self):
        """Testing the function _get_iter_from_Q by comparing the first array of Q and the q_init, which should be the first array of Q 
        """
        self.assertTrue(np.array_equal(self._QP._get_q_iter_from_Q(0), self._q_init),
                        msg="q_iter obtained thanks to the method QP._get_q_iter_from_Q differs from the real q_iter")

    def test_get_difference_between_q_iter(self):
        """Testing the function _get_difference_between_q_iter by comparing its result and the difference between q_target and q_init that should be the first and second array of Q
        """
        self.assertTrue(np.array_equal(self._QP._get_difference_between_q_iter(
            0), self._q_inter - self._q_init), msg=" The difference between the q_iter and q_iter+1 is false")

    def test_distance_endeff_target(self):
        """Testing the function _distance_endeff_target by putting the robot at the configuration q_target whence the target was generated and testing whether the distance between the end-effector and the target is equal to 0
        """
        self.assertAlmostEqual(np.linalg.norm(self._QP._distance_endeff_target(
            self._q_target)), 0, msg="Error while computing the distance between the end effector and the target")
        
    def test_initial_residual(self):
        """Testing whether the initial residual is 0 as it should be
        """
        self.assertTrue(np.array_equal(
            self._initial_residual_handmade, self._initial_residual))

    def test_compute_principal_residual(self):
        """Testing the difference between the residual that was computed in QP and the handmade residual
        """
        self.assertTrue(np.array_equal(self._principal_residual, self._principal_residual_handmade),
                        msg="Error while computing principal residual")

    def test_compute_terminal_residual(self):
        """Testing the difference between the terminal residual that was computed in QP and the handmade residual. As the robot is on the target, the terminal residual ought to be equal to 0
        """
        self.assertEqual(np.linalg.norm(self._QP._terminal_residual),
                         0, msg="Error while computing terminal residual")

    def test_compute_initial_cost(self):
        """Testing the difference between initial cost handmade and computed, should be equal
        """
        self.assertEqual(self._initial_cost_handmade, self._initial_cost,
                         msg="Error while computing the initial cost, should be equal but is not")

    def test_compute_principal_cost(self):
        """Testing the difference between the cost that was computed in QP and the handmade cost
        """
        self.assertEqual(self._principal_cost, self._principal_cost_handmade,
                         msg="Error while computing the principal cost")

    def test_compute_cost(self):
        """As the robot is on the target and has an initial position at the initial configuration, the terminal cost ought to be equal to 0, then the total cost is equal to the principal cost
        """
        self.assertEqual(self._principal_cost_handmade, self._cost,
                         msg="Error while computing the total cost")

    def test_residual_derivative_finite_difference(self):
        """Testing the residual function with the finite difference method defined before
        """
        fdm_res = self._numdiff(self._QP.compute_residuals, self._QP._Q)
        self.assertAlmostEqual(np.linalg.norm(fdm_res - self._derivative_residuals), 0, places=6,
                               msg="The derivative residual is not the same as the finite difference one")

    def test_gradient_finite_difference(self):
        """Testing the gradient of the cost with the finite difference method
        """
        self.assertAlmostEqual(np.linalg.norm(self._gradient - self._gradient_numdiff), 0,
                               places= 5, msg="The gradient is not the same as the finite difference one")

    def test_hessian_finite_difference(self):
        """Testing the hessian of the cost with the finite difference method
        """
        self.assertAlmostEqual(np.linalg.norm(self._hessval - self._hessval_numdiff), 0,
                               places=5, msg=" The hessian is not the same as the finite difference one")

        # Methods used for the tests
    def _numdiff(self, f, x, eps=1e-6):
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
        for i, v in enumerate(x):
            xc[i] += eps
            res.append(copy.copy(f(xc)-f0)/eps)
            xc[i] = x[i]
        return np.array(res).T



if __name__ == "__main__":
    # Start of the unit tests
    unittest.main()
