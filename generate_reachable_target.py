import pinocchio as pin
import numpy as np


def generateReachableTarget(rmodel,rdata = None, frameName = 'endeff'):
    '''
    Sample a random configuration, then returns the forward kinematics
    for this configuration rdata.oMf[frameId].
    If rdata is None, create it on the flight (warning emitted)
    '''
    q_target = pin.randomConfiguration(rmodel)

    # Creation of a temporary model.Data, to have access to the forward kinematics.
    if rdata is None:
        rdata = rmodel.createData()
        print('Warning: pin.Data create for a simple kinematic, please avoid' )

    # Updating the model.Data with the framesForwardKinematics
    pin.framesForwardKinematics(rmodel, rdata, q_target)

    # Get and check Frame Id
    fid = rmodel.getFrameId(frameName)
    assert(fid<len(rmodel.frames))
    
    return rdata.oMf[fid].copy()

if __name__ == "__main__":
    import example_robot_data as robex
    robot = robex.load('ur10')
    p = generateReachableTarget(robot.model,robot.data, 'tool0')

    assert(np.all(np.isfinite(p.translation)))
    
