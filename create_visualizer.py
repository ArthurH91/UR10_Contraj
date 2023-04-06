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


import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import meshcat


def create_visualizer(robot, grid=False, axes=False):
    """ Create a visualizer using meshcat, allowing a robot to be visualized. 

    Parameters
    ----------
    robot : 
        Robot unwrapped from pinocchio
    grid : bool, optional
        Whether the user wants the grid, by default False
    axes : bool, optional
        Whether the user wants the axes, by default False

    Returns
    -------
    vis
        A vis object, can be updated with vis.dispay(q), where q is an np.ndarray of robot.nq dimensions.
    """
    Viewer = pin.visualize.MeshcatVisualizer
    vis = Viewer(robot.model, robot.collision_model, robot.visual_model)
    vis.initViewer(viewer=meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000"))
    vis.viewer.delete()
    vis.loadViewerModel()
    if not grid:
        vis.viewer["/Grid"].set_property("visible", False)
    if not axes:
        vis.viewer["/Axes"].set_property("visible", False)
    return vis

