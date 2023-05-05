import unittest
import numpy as np
from utils import get_q_iter_from_Q, get_transform, get_difference_between_q_iter, generate_reachable_target
import pinocchio as pin


class TestUtils(unittest.TestCase):
    def test_get_iter_from_Q(self):
        """Testing the function _get_iter_from_Q by comparing the first array of Q and the q_init, which should be the first array of Q"""
        q1 = np.ones((6))
        q2 = 2 * np.ones((6))
        Q = np.concatenate((q1, q2))
        self.assertTrue(
            np.array_equal(get_q_iter_from_Q(Q, 0, 6), q1),
            msg="q_iter obtained thanks to the method QP._get_q_iter_from_Q differs from the real q_iter",
        )
        self.assertTrue(
            np.array_equal(get_q_iter_from_Q(Q, 1, 6), q2),
            msg="q_iter obtained thanks to the method QP._get_q_iter_from_Q differs from the real q_iter",
        )

    def test_get_difference_between_q_iter(self):
        """Testing the function get_difference_between_q"""

        q1 = np.ones((6))
        q2 = 2 * np.ones((6))
        Q = np.concatenate((q1, q2))
        self.assertTrue(
            np.array_equal(get_difference_between_q_iter(Q, 0, 6), np.ones(6)),
            msg="Error while doing the difference between the arrays",
        )

    def test_get_transform(self):
        """Testing the function get_transform"""
        T = pin.SE3.Random()
        T_test = get_transform(T)
        self.assertTrue(
            np.all(np.isfinite(T_test)),
            msg="get_transform does not return a finite array",
        )
        self.assertTrue(
            np.array_equal(T_test[:3, 3], T.translation),
            msg=" Problem while comparing translation parts",
        )
        self.assertTrue(
            np.array_equal(T_test[:3, :3], T.rotation),
            msg="Problem while comparing the rotation parts",
        )

    def test_generate_reachable_target(self):
        """Testing the function generate_reachable_target by making sure it returns a finite array.
        """
        import example_robot_data as robex
        robot = robex.load("ur10")
        p = generate_reachable_target(robot.model, robot.data, "tool0")
        self.assertTrue(np.all(np.isfinite(p.translation)))

if __name__ == "__main__":
    unittest.main()
