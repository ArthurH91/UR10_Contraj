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
import time

from RobotWrapper import RobotWrapper
from create_visualizer import create_visualizer
from QuadraticProblemNLP import QuadratricProblemNLP
from NewtonMethodMarcToussaint import NewtonMethodMt

def callback(q):
    vis.display(q)
    time.sleep(1e-3)
    input()

if __name__ == "__main__":


    # Creation of the robot
    robot_wrapper = RobotWrapper()
    robot, rmodel, gmodel = robot_wrapper(target=True)
    rdata = rmodel.createData()
    gdata = gmodel.createData()

    # Open the viewer
    vis = create_visualizer(robot)

    # Creating the QP 
    T = 4
    QP = QuadratricProblemNLP(rmodel, rdata, gmodel, gdata, T, k1 = 1, k2=10 )

    # Initial configuration
    # pin.seed(0)
    q0 = pin.randomConfiguration(rmodel)
    robot.q0 = q0

    vis.display(q0)
    Q = np.array(q0)
    for i in range(T-1):
        Q = np.concatenate((Q, q0))

    eps = 1e-5

    Solv = NewtonMethodMt(QP.compute_cost, QP.grad, QP.hess, max_iter = 10, callback=callback)

    res = Solv(Q)