import numpy as np

from Data.UE4Data import convert_data
from socket_io_client import SocketIoClient
from Transformations import cv_lib
from Transformations.transform import Transform, Scaling
from scipy.spatial.transform import Rotation as R

if __name__ == '__main__':
    # ================================================================================================================
    # Connect to UE4 Server
    # ================================================================================================================
    sio = SocketIoClient()

    # ================================================================================================================
    # Extract UE4 Data (LEFT-HAND)
    # ================================================================================================================
    data = None
    while data is None:  # Waiting for ue4 to connect to server:
        data = sio.requestUE4Data()
    drone_data, ship_data = convert_data(data)
    assert drone_data is not None
    assert ship_data is not None
    drone_location = drone_data.location
    drone_rotation = drone_data.rotation
    ship_location = ship_data.location
    ship_rotation = ship_data.rotation

    # ================================================================================================================
    # Find left-hand to right-hand system:
    # ================================================================================================================
    # Very simple Notes: https://www.tutorialspoint.com/computer_graphics/3d_transformation.htm
    # Change direction of y axis (by scaling)
    scaling = Scaling(y=-1).m_scaling
    print(scaling)

    # Find the rotation between left-hand coordinates to right-hand coordinates
    _, l_to_r_rotation = cv_lib.get_3d_rotation_matrix_from_yaw_pitch_roll(
        yaw=-90.0,
        pitch=0.0,
        roll=0.0,
        degrees=True,
        verbose=False)

    t_left_to_right = Transform(rotation=l_to_r_rotation, scaling=scaling)
    # Since the right handed coordinate system is using reverse rotation, we do not need to inverse the transformation
    t_right_to_left = t_left_to_right

    # Convert incomming rotations:
    rot_transform = {'yaw_trans': -1, 'pitch_trans': 1, 'roll_trans': -1}


    def rotation_transform(yaw, pitch, roll):
        y = yaw * rot_transform['yaw_trans']
        p = roll * rot_transform['roll_trans']
        r = pitch * rot_transform['pitch_trans']
        return y, p, r


    # ================================================================================================================
    # Check if true:
    # ================================================================================================================
    pos_left = np.array([1., 2., 3.]).reshape((3, 1))
    pos_left_in_right = np.array([2., 1., 3.]).reshape((3, 1))

    l_to_r_pos = t_left_to_right(pos_left)
    assert np.equal(l_to_r_pos, pos_left_in_right).all()
    r_to_l_pos = t_right_to_left(l_to_r_pos)
    assert np.equal(r_to_l_pos, pos_left).all()

    debug = 0
    # ================================================================================================================
    # Find world to Drone transform_matrix
    # ================================================================================================================
    # Find the rotation between drone and world axis (The difference in (Drone rotation) and world rotation = 0,0,0)
    d_yaw_r, d_pitch_r, d_roll_r = rotation_transform(drone_data.rotation.yaw,
                                                      drone_data.rotation.pitch,
                                                      drone_data.rotation.roll)

    _, d_rotation_r = cv_lib.get_3d_rotation_matrix_from_yaw_pitch_roll(
        yaw=d_yaw_r,
        pitch=d_pitch_r,
        roll=d_roll_r,
        degrees=True,
        verbose=False)

    r = R.from_matrix(d_rotation_r.T).as_euler('zyx', degrees=True)
    # Find the translation between drone coord to world coord ( Since the location we have is in world coord its easy)
    d_location_l = np.array([drone_data.location.x, drone_data.location.y, drone_data.location.z])
    d_location_r = t_left_to_right(d_location_l)

    # Combine into transformation matrix from drone coord to world coord
    t_drone_to_world = Transform(d_location_r, d_rotation_r)

    # To find the inverse transformation we invert the matrix ( Hence now we have from world to drone transformation)
    t_world_to_drone = t_drone_to_world.inverse()

    # ================================================================================================================
    # Check if true:
    # ================================================================================================================
    # The zero test
    drone_world_pos_l = d_location_l
    drone_world_pos_r = t_left_to_right(drone_world_pos_l)
    drone_drone_pos_r = t_world_to_drone(drone_world_pos_r)
    drone_drone_pos_l = t_right_to_left(drone_drone_pos_r)

    # the ship test:
    ship_world_pos_l = np.array([ship_data.location.x, ship_data.location.y, ship_data.location.z]).round(decimals=12)
    ship_world_pos_r = t_left_to_right(ship_world_pos_l)
    ship_drone_pos_r = t_world_to_drone(ship_world_pos_r)
    ship_drone_pos_l = t_right_to_left(ship_drone_pos_r)

    # the ship drone to world test:

    ship_drone_pos_l = np.array([355.552124, 353.676727, -488.427063])
    ship_drone_pos_r = t_left_to_right(ship_drone_pos_l)
    ship_world_pos_r = t_drone_to_world(ship_drone_pos_r)
    ship_world_pos_l = t_right_to_left(ship_world_pos_r)

    debug = 0
