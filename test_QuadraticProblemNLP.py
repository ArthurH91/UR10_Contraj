import unittest

import numpy as np
import pinocchio as pin
import copy

from QuadraticProblemNLP import QuadratricProblemNLP
from RobotWrapper import RobotWrapper


class TestQuadraticProblemNLP(unittest.TestCase):

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
        print(x)
        xc = np.copy(x)
        f0 = np.copy(f(x))
        res = []
        for i in range(len(x)):
            xc[i] += eps
            res.append(copy.copy(f(xc)-f0)/eps)
            xc[i] = x[i]
        print(f"res = {res}")
        return np.array(res).T


    # Tests 

    def test_get_q_iter_from_Q(self):
        """Testing the function _get_iter_from_Q by comparing the first array of Q and the q_init, which should be the first array of Q 
        """
        self.assertTrue(np.array_equal(QP._get_q_iter_from_Q(0), q_init),
                        msg="q_iter obtained thanks to the method QP._get_q_iter_from_Q differs from the real q_iter")

    def test_get_difference_between_q_iter(self):
        """Testing the function _get_difference_between_q_iter by comparing its result and the difference between q_target and q_init that should be the first and second array of Q
        """
        self.assertTrue(np.array_equal(QP._get_difference_between_q_iter(0), q_target - q_init), msg=" The difference between the q_iter and q_iter+1 is false")

    def test_distance_endeff_target(self):
        """Testing the function _distance_endeff_target by putting the robot at the configuration q_target whence the target was generated and testing whether the distance between the end-effector and the target is equal to 0
        """
        self.assertAlmostEqual(np.linalg.norm(QP._distance_endeff_target(
            q_target)), 0, msg="Error while computing the distance between the end effector and the target")
        
    def test_compute_principal_residual(self):
        """Testing the difference between the residual that was computed in QP and the handmade residual
        """
        self.assertTrue(np.array_equal(principal_residual, residual_handmade), msg = "Error while computing principal residual")

    def test_compute_terminal_residual(self):
        """Testing the difference between the terminal residual that was computed in QP and the handmade residual. As the robot is on the target, the terminal residual ought to be equal to 0
        """
        self.assertEqual(np.linalg.norm(QP._terminal_residual), 0, msg = "Error while computing terminal residual")

    def test_compute_principal_cost(self):
        """Testing the difference between the cost that was computed in QP and the handmade cost
        """
        self.assertEqual(principal_cost, principal_cost_handmade, msg = "Error while computing the principal cost")

    def test_compute_cost(self):
        """As the robot is on the target, the terminal cost ought to be equal to 0, then the total cost is equal to the principal cost
        """
        self.assertEqual(principal_cost_handmade, cost, msg = "Error while computing the total cost")

    def test_compute_derivative_principal_residuals_filled_part(self):
        """Testing the function _compute_derivative_principal_residuals, as the matrix is composed of 2 identity matrixes where one is one line under the diagonal, the determinant should be equal to 1.
        Testing only the identity block matrix, not the whole one because computing the determinant is only feasible for a squared matrix
        """
        self.assertEqual(np.linalg.det(
            derivative_principal_residuals[:rmodel.nq, :rmodel.nq]), QP._k1, msg="The determinant of the principal residual is not equal to 1")

    def test_compute_derivative_principal_residuals_terminal_residual_part(self):
        """Testing the function _compute_derivative_principal_residuals, verifying that the part of the matrix reserved for the terminal residual derivates is null
        """
        self.assertTrue(np.array_equal(
            np.zeros((3, 6)), derivative_principal_residuals[-3:, -6:]), msg= "The terminal residual part of the derivatives of the principal residual should be null but is not")

    def test_compute_derivative_terminal_residuals(self):
        """I did not found a method to test this part of the residual yet
        """
        pass

    # def test_grad(self):
    #     """Testing the grad function with the finite difference method define before
    #     """
    #     grad_numdiff = self._numdiff(QP.grad, QP._Q)
    #     self.assertTrue(np.isclose(grad_numdiff, gradient), msg= "The gradient is not the same as the finite difference one")


    
if __name__ == "__main__":

    # Setup of the environement
    robot_wrapper = RobotWrapper()
    robot, rmodel, gmodel = robot_wrapper(target=True)
    rdata = rmodel.createData()
    gdata = gmodel.createData()
    QP = QuadratricProblemNLP(rmodel, rdata, gmodel, gdata, T=2)

    # Variables for the tests of _get_q_iter_from_Q and _distance_endeff_target
    q_target = robot_wrapper._q_target  # Configuration array reaching the target
    q_init = pin.randomConfiguration(rmodel)
    Q_target = np.concatenate((q_init, q_target))
    QP._Q = Q_target

    # Variables for the test_compute_residuals
    residual_handmade = q_target - q_init
    residual = QP.compute_residuals(Q_target)
    principal_residual = QP._principal_residual

    # Variables for computing the cost 
    cost = QP.compute_cost()
    principal_cost = QP._principal_cost
    principal_cost_handmade = 0.5 * np.linalg.norm(q_target - q_init) ** 2

    # Variables for computing the derivatives of the residuals
    derivative_principal_residuals = QP._compute_derivative_principal_residuals()

    # Variables for computing the gradient 
    gradient = QP.grad(Q_target)

    # Start of the unit tests
    unittest.main()
