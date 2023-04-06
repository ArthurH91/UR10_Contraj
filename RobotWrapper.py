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
import example_robot_data as robex
from create_visualizer import create_visualizer
import hppfcl

# This class is for unwrapping an URDF and converting it to a model. It is also possible to add objects in the model,
# such as a ball at a specific position.

class RobotWrapper():
    def __init__(self, scale=1.0, name_robot="ur10"):
        """Initialize the wrapper with a scaling number of the target and the name of the robot wanted to get unwrapped.

        Parameters
        ----------
        _scale : float, optional
            Scale of the target, by default 1.0
        name_robot : str, optional
            Name of the robot wanted to get unwrapped, by default "ur10"
        """

        self._scale = scale
        self._robot = robex.load(name_robot)
        self._rmodel = self._robot.model
        self._color = np.array([249, 136, 126, 255]) / 255

    def __call__(self, target=False):
        """Create a robot with a new frame at the end effector position and place a hppfcl: ShapeBase cylinder at this position.

        Parameters
        ----------
        target : bool, optional
            Boolean describing whether the user wants a target or not, by default False

        Returns
        -------
        _robot
            Robot description of the said robot
        _rmodel
            Model of the robot
        _gmodel
            Geometrical model of the robot


        """

        # Creation of the frame for the end effector by using the frame tool0, which is at the end effector pose.
        # This frame will be used for the position of the cylinder at the end of the effector.
        # The cylinder is used to have a HPPFCL shape at the end of the robot to make contact with the target

        # Obtaining the frame ID of the frame tool0
        ID_frame_tool0 = self._rmodel.getFrameId('tool0')
        # Obtaining the frame tool0
        frame_tool0 = self._rmodel.frames[ID_frame_tool0]
        # Obtaining the parent joint of the frame tool0
        parent_joint = frame_tool0.parentJoint
        # Obtaining the placement of the frame tool0
        Mf_endeff = frame_tool0.placement
        # Creating the endeff frame
        endeff_frame = pin.Frame("endeff", parent_joint, Mf_endeff, pin.BODY)
        _ = self._rmodel.addFrame(endeff_frame, False)

        # Creation of the geometrical model
        self._gmodel = self._robot.visual_model

        # Creation of the cylinder at the end of the end effector

        # Setting up the raddi of the cylinder
        endeff_radii, endeff_width = 1e-2, 1e-2
        # Creating a HPPFCL shape
        endeff_shape = hppfcl.Cylinder(endeff_radii, endeff_width)
        # Creating a pin.GeometryObject for the model of the _robot
        geom_endeff = pin.GeometryObject(
            "endeff_geom", parent_joint, Mf_endeff, endeff_shape)
        geom_endeff.meshColor = self._color
        # Add the geometry object to the geometrical model
        self._gmodel.addGeometryObject(geom_endeff)

        if target:
            self._create_target()

        return self._robot, self._rmodel, self._gmodel

    def _create_target(self):
        """ Updates the version of the robot models with a sphere that can be used as a target.

        Returns
        -------
        _robot
            Robot description of the said robot
        _rmodel
            Model of the robot
        _gmodel 
            Geometrical model of the robot
        """

        # Setup of the shape of the target (a sphere here)
        r_target = 5e-2*self._scale

        # Creation of the target

        # Creating the frame of the target

        self._M_target = self._generate_reachable_SE3_vector()

        target_frame = pin.Frame("target", self._rmodel.getJointId(
            "universe"), self._M_target, pin.BODY)
        target = self._rmodel.addFrame(target_frame, False)
        T_target = self._rmodel.frames[target].placement
        target_shape = hppfcl.Sphere(r_target)
        geom_target = pin.GeometryObject("target_geom", self._rmodel.getJointId(
            "universe"), T_target, target_shape)

        geom_target.meshColor = self._color
        self._gmodel.addGeometryObject(geom_target)


    def _generate_reachable_SE3_vector(self):
        """ Generate a SE3 vector that can be reached by the robot.

        Returns
        -------
        Reachable_SE3_vector
            SE3 Vector describing a reachable position by the robot
        """
        
        # Generate a random configuration of the robot, with consideration to its limits
        self._q_target = pin.randomConfiguration(self._rmodel)
        # Creation of a temporary model.Data, to have access to the forward kinematics.
        ndata = self._rmodel.createData()
        # Updating the model.Data with the framesForwardKinematics
        pin.framesForwardKinematics(self._rmodel, ndata, self._q_target)

        return ndata.oMf[self._rmodel.getFrameId('endeff')]



if __name__ == "__main__":

    robot_wrapper = RobotWrapper()
    robot, rmodel, gmodel = robot_wrapper(target=True)
    vis = create_visualizer(robot)
