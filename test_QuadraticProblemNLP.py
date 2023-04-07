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
        residual = QP.compute_residuals(Q_target)
        principal_residual = QP._principal_residual
        self.assertTrue(np.array_equal(principal_residual, residual_handmade), msg = "Error while computing principal residual")

    


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

    # Start of the unit tests
    unittest.main()
