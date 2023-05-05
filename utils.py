import hppfcl
import pinocchio as pin
import numpy as np    
import time

def get_transform(T_: hppfcl.Transform3f):
    T = np.eye(4)
    if isinstance(T_, hppfcl.Transform3f):
        T[:3, :3] = T_.getRotation()
        T[:3, 3] = T_.getTranslation()
    elif isinstance(T_, pin.SE3):
        T[:3, :3] = T_.rotation
        T[:3, 3] = T_.translation
    else:
        raise NotADirectoryError
    return T

def get_q_iter_from_Q(Q : np.ndarray, iter: int, nq: int):
    """Returns the iter-th configuration vector q_iter in the Q array.

        Args:
            Q (np.ndarray): Optimization vector.
            iter (int): Index of the q_iter desired.
            nq (int): size of q_iter

        Returns:
            q_iter (np.ndarray): Array of the configuration of the robot at the iter-th step.
        """
    q_iter = np.array((Q[nq * iter: nq * (iter+1)]))
    return q_iter


def get_difference_between_q_iter(Q: np.ndarray, iter: int, nq: int):
    """Returns the difference between the q_iter and q_iter+1 in the array self.Q

    Parameters
    ----------
    Q : np.ndarray 
        Optimization vector.
    iter : int
        Index of the q_iter desired.
    nq : int
        Length of a configuration vector.
    
    Returns:
        q_iter+1 - q_iter (np.ndarray): Difference of the arrays of the configuration of the robot at the iter-th and ither +1 -th steps.

    """
    return get_q_iter_from_Q(Q, iter + 1, nq) - get_q_iter_from_Q(Q, iter, nq)

def display_last_traj(vis, Q: np.ndarray, q0: np.ndarray, T : int, dt = None):
    """Display the trajectory computed by the solver

    Parameters
    ----------
    vis : Meshcat.Visualizer
        Meshcat visualizer
    Q : np.ndarray
        Optimization vector.
    q0 : np.ndarray
        Initial configuration vector
    nq : int
        size of q_iter
    """
    for q_iter in [q0]+np.split(Q,T+1):
        vis.display(q_iter)
        if dt is None:
            input()
        else:
            time.sleep(dt)



    