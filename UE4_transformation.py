import numpy as np

from Data.UE4Data import convert_data, State
from socket_io_client import SocketIoClient
from Transformations import cv_lib
from Transformations.transform import Transform, Scaling, Rotation
from scipy.spatial.transform import Rotation as R

from test import gt_transform

camera_focal_length = 0
camera_fov = 0

camera_state_in_drone_coord = State({'x': 0.0, 'y': 0.0, 'z': -5.0},
                                    {'pitch': -90, 'roll': 0, 'yaw': 0})


def get_drone_to_camera_transformation():
    # ================================================================================================================
    # Find drone to camera transform_matrix
    # ================================================================================================================
    # Since the camera information is static atm we just use static values:
    global camera_state_in_drone_coord
    rotation = camera_state_in_drone_coord.rotation
    location = camera_state_in_drone_coord.location
    # Find the rotation between camera and drone axis

    camera_rotation_x = Rotation(rotation.roll, axis='x', degrees=True, left_hand=False)
    camera_rotation_y = Rotation(rotation.pitch, axis='y', degrees=True, left_hand=False)
    camera_rotation_z = Rotation(rotation.yaw, axis='z', degrees=True, left_hand=True)
    camera_rotation_xy = camera_rotation_x.after(camera_rotation_y)
    camera_rotation_xyz = camera_rotation_xy.after(camera_rotation_z)

    # Find the translation between camera coord to drone coord
    camera_pos_d = np.array([location.x, location.y, location.z]).reshape(3, 1)
    drone_to_camera_trans = - camera_pos_d

    # Invert the translation
    drone_to_camera = Transform(rotation=camera_rotation_xyz,
                                translation=drone_to_camera_trans,
                                translate_before_rotate=True)
    camera_to_drone = world_to_drone.inverse()

    return drone_to_camera, camera_to_drone


def find_front_offset(ship_length):
    if 0 < ship_length < 100:
        return - (ship_length * 0.07)
    if ship_length < 150:
        return -(ship_length * 0.10)
    if ship_length < 200:
        return -(ship_length * 0.13)
    return -(ship_length * 0.16)


def compute_ship_model_points(dist_to_bow: int,
                              dist_to_stern: int,
                              dist_to_starboard: int,
                              dist_to_port: int):
    """
    :param dist_to_bow: distance from gps antenna to bow(front) (meters)
    :param dist_to_stern: distance from gps antenna to stern(back) (meters)
    :param dist_to_starboard: distance from gps antenna to starboard(right) (meters)
    :param dist_to_port: distance from gps antenna to port(left) (meters)
    :return: a dictionary with points
    """
    ship_length = dist_to_bow + dist_to_stern
    p_front_center = np.array([dist_to_bow, 0, 0])
    p_back_center = np.array([-dist_to_stern, 0, 0])
    p_right_center = np.array([0, dist_to_starboard, 0])
    p_left_center = np.array([0, -dist_to_port, 0])

    front_offset = np.array([find_front_offset(ship_length), 0, 0])

    p_back_left_corner = p_back_center + p_left_center
    p_back_right_corner = p_back_center + p_right_center
    p_front_left_corner = p_front_center + front_offset + p_left_center
    p_front_right_corner = p_front_center + front_offset + p_right_center

    result = {
        'p_front_center': p_front_center,
        'p_front_right_corner': p_front_right_corner,
        'p_front_left_corner': p_front_left_corner,
        'p_back_right_corner': p_back_right_corner,
        'p_back_left_corner': p_back_left_corner
    }
    return result


if __name__ == '__main__':
    # ================================================================================================================
    # Connect to UE4 Server
    # ================================================================================================================
    sio = SocketIoClient()
    staticShipData = None
    ship_model_points_l = None
    drone_to_camera, camera_to_drone = get_drone_to_camera_transformation()
    # ================================================================================================================
    # Extract UE4 Data (LEFT-HAND axis :: Rotations = roll: Right-handed, pitch: Right-handed, yaw: Left-handed)
    # ================================================================================================================
    data = None
    while data is None:  # Waiting for ue4 to connect to server:
        data = sio.requestUE4Data()
    drone_data, ship_data = convert_data(data)
    assert drone_data is not None
    assert ship_data is not None

    if staticShipData is None:
        staticShipData = data['StaticShipData']  # [front,back,left,right]
        ship_model_points_l = compute_ship_model_points(staticShipData[0],
                                                        staticShipData[1],
                                                        staticShipData[3],
                                                        staticShipData[2])

    drone_location = drone_data.location
    drone_rotation = drone_data.rotation
    ship_location = ship_data.location
    ship_rotation = ship_data.rotation

    # ================================================================================================================
    # Find world to Drone transform_matrix
    # ================================================================================================================
    # Find the rotation between drone and world axis (The difference in (Drone rotation) and world rotation = 0,0,0)
    drone_rotation_x = Rotation(drone_rotation.roll, axis='x', degrees=True, left_hand=False)
    drone_rotation_y = Rotation(drone_rotation.pitch, axis='y', degrees=True, left_hand=False)
    drone_rotation_z = Rotation(drone_rotation.yaw, axis='z', degrees=True, left_hand=True)
    drone_rotation_xy = drone_rotation_x.after(drone_rotation_y)
    drone_rotation_xyz = drone_rotation_xy.after(drone_rotation_z)

    # Find the translation between drone coord to world coord ( Since the location we have is in world coord its easy)
    drone_pos_w = np.array([drone_location.x, drone_location.y, drone_location.z]).reshape(3, 1)
    world_to_drone_trans = - drone_pos_w

    # Invert the translation
    world_to_drone = Transform(rotation=drone_rotation_xyz,
                               translation=world_to_drone_trans,
                               translate_before_rotate=True)
    drone_to_world = world_to_drone.inverse()

    # ================================================================================================================
    # Find world to Ship transform_matrix
    # ================================================================================================================
    # Find the rotation between Ship and world axis
    ship_rotation_x = Rotation(ship_rotation.roll, axis='x', degrees=True, left_hand=False)
    ship_rotation_y = Rotation(ship_rotation.pitch, axis='y', degrees=True, left_hand=False)
    ship_rotation_z = Rotation(ship_rotation.yaw, axis='z', degrees=True, left_hand=True)
    ship_rotation_xy = ship_rotation_x.after(ship_rotation_y)
    ship_rotation_xyz = ship_rotation_xy.after(ship_rotation_z)

    # Find the translation between Ship coord to world coord ( Since the location we have is in world coord its easy)
    ship_pos_w = np.array([ship_location.x, ship_location.y, ship_location.z]).reshape(3, 1)
    world_to_ship_trans = - ship_pos_w

    # Invert the translation
    world_to_ship = Transform(rotation=ship_rotation_xyz,
                              translation=world_to_ship_trans,
                              translate_before_rotate=True)
    ship_to_world = world_to_drone.inverse()

    debug = 0
