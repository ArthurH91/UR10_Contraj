import unittest

import numpy as np
import pinocchio as pin

from QuadraticProblemNLP import QuadratricProblemNLP
from RobotWrapper import RobotWrapper


class TestQuadraticProblemNLP(unittest.TestCase):

    def test_get_q_iter_from_Q(self):
        self.assertTrue(np.array_equal(QP._get_q_iter_from_Q(0), q_init),
                        msg="q_iter obtained thanks to the method QP._get_q_iter_from_Q differs from the real q_iter")

    def test_get_difference_between_q_iter(self):
        self.assertTrue(np.array_equal(QP._get_q_iter_from_Q(1) - QP._get_q_iter_from_Q(
            0), q_target - q_init), msg=" The difference between the q_iter and q_iter+1 is false")

    def test_distance_endeff_target(self):
        self.assertAlmostEqual(np.linalg.norm(QP._distance_endeff_target(
            q_target)), 0, msg="Error while computing the distance between the end effector and the target")
        
    def test_compute_principal_residual(self):
        self.assertTrue(np.array_equal(principal_residual, residual_handmade), msg = "Error while computing principal residual")

    def test_compute_terminal_residual(self):
        self.assertEqual(np.linalg.norm(QP._terminal_residual), 0, msg = "Error while computing terminal residual")

    def test_compute_principal_cost(self):
        self.assertEqual(principal_cost, principal_cost_handmade, msg = "Error while computing the principal cost")

    def test_compute_cost(self):
        """As the robot is on the target, the terminal cost ought to be equal to 0, then the total cost is equal to the principal cost
        """
        self.assertEqual(principal_cost_handmade, cost, msg = "Error while computing the total cost")
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


    # Start of the unit tests
    unittest.main()
