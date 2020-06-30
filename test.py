import numpy as np

from Data.UE4Data import convert_data, State
from socket_io_client import SocketIoClient
from Transformations import cv_lib
from Transformations.transform import Vector, Rotation, Transform

ship_length = 175.50
ship_width = 40.00

camera_focal_length = 0
camera_fov = 0

camera_state_in_drone_coord = State({'x': 0.0, 'y': 0.0, 'z': -5.0},
                                    {'pitch': -90, 'roll': 0, 'yaw': 0})  # TODO: Why is 90 working but not -90


def create_top_view_ship_camera(altitude):
    ship_pixel_width = ship_width / altitude
    ship_pixel_length = ship_length / altitude


def gt_transform(gt_tf, decimals=2):
    gt_trans = np.array([gt_tf['wPlane']['x'], gt_tf['wPlane']['y'],
                         gt_tf['wPlane']['z']]).reshape(3, 1)
    gt_xplane = np.array([gt_tf['xPlane']['x'], gt_tf['xPlane']['y'],
                          gt_tf['xPlane']['z']]).reshape(3, 1)
    gt_yplane = np.array([gt_tf['yPlane']['x'], gt_tf['yPlane']['y'],
                          gt_tf['yPlane']['z']]).reshape(3, 1)
    gt_zplane = np.array([gt_tf['zPlane']['x'], gt_tf['zPlane']['y'],
                          gt_tf['zPlane']['z']]).reshape(3, 1)
    gt_rot = np.append(np.append(gt_xplane, gt_yplane), gt_zplane).reshape(3, 3)
    t_gt = Transform(gt_trans, gt_rot)
    gt = np.hstack((gt_rot, gt_trans))
    gt = np.vstack((gt, [0, 0, 0, 1]))
    gt = np.round(gt, decimals=decimals)
    return gt, t_gt


if __name__ == '__main__':

    # ================================================================================================================
    # Connect to UE4 Server
    # ================================================================================================================
    sio = SocketIoClient()

    # ================================================================================================================
    # Extract UE4 Data
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
    # TEST
    # Change direction of y-axis
    scale_matrix = np.identity(4)
    scale_matrix[1][1] = -scale_matrix[1][1]

    ship_w = np.array([638.257202, 780.391113, 563.965942])
    ship_d = np.array([353.550720, 353.546204, 489.989777])
    ship_c = np.array([484.990356, 353.545746, 353.550629])
    drone_w = np.array([0, 500, 500])
    drone_c = np.array([0, 0, 5])

    # ship_w = np.array([1, 1, 1])
    # ship_d = np.array([1, 1, 1])
    #
    # drone_w = np.array([0, 1, 1])

    v_drone_w = Vector(drone_w)
    v_ship_w = Vector(ship_w)
    v_ship_d = Vector(ship_d)

    drone_rotate, drone_rotation_matrix = cv_lib.get_3d_rotation_matrix_from_yaw_pitch_roll(-drone_data.rotation.yaw,
                                                                                            -drone_data.rotation.pitch,
                                                                                            -drone_data.rotation.roll,
                                                                                            degrees=True)

    r_drone_w = Rotation().set_matrix(drone_rotation_matrix)

    drone_to_world = Transform(v_drone_w, r_drone_w)
    world_to_drone = drone_to_world.inverse()

    v1 = world_to_drone.apply_to(v_drone_w)
    v1_2 = drone_to_world.apply_to(v_drone_w)
    v2 = world_to_drone.apply_to(v_ship_w)
    v2_2 = drone_to_world.apply_to(v_ship_d)

    # [drone_data.rotation.yaw, drone_data.rotation.pitch, drone_data.rotation.roll,
    #  drone_data.location.x, drone_data.location.y, drone_data.location.z]

    CONF = [[1, 1, 1,
             1, 1, 1],

            [-1, 1, 1,
             1, 1, 1],

            [1, -1, 1,
             1, 1, 1],

            [1, 1, -1,
             1, 1, 1],

            [-1, -1, 1,
             1, 1, 1],

            [1, -1, -1,
             1, 1, 1],

            [-1, -1, -1,
             1, 1, 1],

            [1, 1, 1,
             - 1, 1, 1],

            [-1, 1, 1,
             -1, 1, 1],

            [1, -1, 1,
             -1, 1, 1],

            [1, 1, -1,
             -1, 1, 1],

            [-1, -1, 1,
             -1, 1, 1],

            [1, -1, -1,
             -1, 1, 1],

            [-1, -1, -1,
             -1, 1, 1],

            [1, 1, 1,
             1, -1, 1],

            [-1, 1, 1,
             1, -1, 1],

            [1, -1, 1,
             1, -1, 1],

            [1, 1, -1,
             1, -1, 1],

            [-1, -1, 1,
             1, -1, 1],

            [1, -1, -1,
             1, -1, 1],

            [-1, -1, -1,
             1, -1, 1],

            [1, 1, 1,
             1, 1, -1],

            [-1, 1, 1,
             1, 1, -1],

            [1, -1, 1,
             1, 1, -1],

            [1, 1, -1,
             1, 1, -1],

            [-1, -1, 1,
             1, 1, -1],

            [1, -1, -1,
             1, 1, -1],

            [-1, -1, -1,
             1, 1, -1],

            [1, 1, 1,
             -1, -1, 1],

            [-1, 1, 1,
             -1, -1, 1],

            [1, -1, 1,
             -1, -1, 1],

            [1, 1, -1,
             -1, -1, 1],

            [-1, -1, 1,
             -1, -1, 1],

            [1, -1, -1,
             -1, -1, 1],

            [-1, -1, -1,
             -1, -1, 1],

            [1, 1, 1,
             -1, 1, -1],

            [-1, 1, 1,
             -1, 1, -1],

            [1, -1, 1,
             -1, 1, -1],

            [1, 1, -1,
             -1, 1, -1],

            [-1, -1, 1,
             -1, 1, -1],

            [1, -1, -1,
             -1, 1, -1],

            [-1, -1, -1,
             -1, 1, -1],

            [1, 1, 1,
             1, -1, -1],

            [-1, 1, 1,
             1, -1, -1],

            [1, -1, 1,
             1, -1, -1],

            [1, 1, -1,
             1, -1, -1],

            [-1, -1, 1,
             1, -1, -1],

            [1, -1, -1,
             1, -1, -1],

            [-1, -1, -1,
             1, -1, -1]
            ]

    gt_wl = gt_transform(drone_data.world_to_local)
    gt_lw = gt_transform(drone_data.local_to_world)

    for conf in CONF:
        # [drone_data.rotation.yaw, drone_data.rotation.pitch, drone_data.rotation.roll,
        #  drone_data.location.x, drone_data.location.y, drone_data.location.z]
        yaw = conf[0] * drone_data.rotation.yaw
        pitch = conf[1] * drone_data.rotation.pitch
        roll = conf[2] * drone_data.rotation.roll
        x = conf[3] * drone_data.location.x
        y = conf[4] * drone_data.location.y
        z = conf[5] * drone_data.location.z
        if conf[0] == -1 and conf[4] == 1 and conf[5] == 1:
            debug = 0

        # ================================================================================================================
        # Find world to Drone transform_matrix
        # ================================================================================================================
        # Find the rotation between drone and world axis (The difference in (Drone rotation) and world rotation = 0,0,0)
        drone_rotate, drone_rotation_matrix = cv_lib.get_3d_rotation_matrix_from_yaw_pitch_roll(
            yaw,
            pitch,
            roll,
            degrees=True,
            verbose=False)
        # Find the translation between drone coord to world coord ( Since the location we have is in world coord its easy)
        drone_translate_vector = np.array([x, y, z])

        # Combine into transformation matrix from drone coord to world coord
        drone_to_world = cv_lib.find_transformation_matrix(drone_translate_vector, drone_rotation_matrix)

        # To find the inverse transformation we invert the matrix ( Hence now we have from world to drone transformation)
        world_to_drone = np.linalg.inv(drone_to_world)
        tmp = np.zeros((4, 1))
        tmp = np.copy(drone_to_world.T[3])
        holder = drone_to_world.T
        holder[3] = world_to_drone.T[3]
        drone_to_world = holder.T
        holder = world_to_drone.T
        holder[3] = tmp
        world_to_drone = holder.T

        drone_to_world = np.round(drone_to_world, decimals=2)
        world_to_drone = np.round(world_to_drone, decimals=2)

        if ((drone_to_world - gt_lw) == np.zeros((4, 4))).all() or (
                (world_to_drone - gt_wl) == np.zeros((4, 4))).all():
            print("Drone_to_world\n", (drone_to_world - gt_lw), "")
            print("world_to_drone\n", (world_to_drone - gt_wl), "\n\n")
            print(conf)
            break
        else:
            print(conf)
            print("Drone_to_world\n", (drone_to_world - gt_lw), "")
            print("world_to_drone\n", (world_to_drone - gt_wl), "\n\n")

    # ================================================================================================================
    # Find world to Drone transform_matrix
    # ================================================================================================================
    # Find the rotation between drone and world axis (The difference in (Drone rotation) and world rotation = 0,0,0)
    drone_rotate, drone_rotation_matrix = cv_lib.get_3d_rotation_matrix_from_yaw_pitch_roll(
        -drone_data.rotation.yaw,
        drone_data.rotation.pitch,
        drone_data.rotation.roll,
        degrees=True,
        verbose=False)

    v_drone_w = Vector((drone_data.location.x, drone_data.location.y, drone_data.location.z))
    r_drone_w = Rotation().set_matrix(drone_rotate.as_matrix())
    t_drone_to_world = Transform(v_drone_w, r_drone_w)
    ti_world_to_drone = t_drone_to_world.inverse()

    # Find the translation between drone coord to world coord ( Since the location we have is in world coord its easy)
    drone_translate_vector = np.array([drone_data.location.x, drone_data.location.y, drone_data.location.z])

    # Combine into transformation matrix from drone coord to world coord
    drone_to_world = cv_lib.find_transformation_matrix(drone_translate_vector, drone_rotation_matrix)

    # To find the inverse transformation we invert the matrix ( Hence now we have from world to drone transformation)
    world_to_drone = np.linalg.inv(drone_to_world)
    tmp = np.zeros((4, 1))
    tmp = np.copy(drone_to_world.T[3])
    holder = np.copy(drone_to_world.T)
    holder[3] = np.copy(world_to_drone.T[3])
    drone_to_world = np.copy(holder.T)
    holder = np.copy(world_to_drone.T)
    holder[3] = np.copy(tmp)
    world_to_drone = np.copy(holder.T)

    # ================================================================================================================
    # Check if true:
    # ================================================================================================================
    drone_world_pos = np.append(drone_translate_vector, 1)
    drone_world_pos = np.reshape(drone_world_pos, (4, 1))

    drone_drone_pos = np.dot(world_to_drone, drone_world_pos)
    drone_drone_pos = np.round(drone_drone_pos, decimals=12)

    ship_translate_vector = np.array([ship_data.location.x, ship_data.location.y, ship_data.location.z, 1])
    ship_translate_vector = np.round(ship_translate_vector, decimals=12)

    ship_coord_in_world = np.reshape(ship_translate_vector, (4, 1))
    ship_in_drone_coord = np.dot(world_to_drone, ship_coord_in_world)
    ship_in_drone_coord = np.round(ship_in_drone_coord, decimals=12)

    zero = np.array([0, 0, 0, 1])
    zero = np.reshape(zero, (4, 1))
    test_inv = np.matmul(drone_to_world, zero)
    test_inv = np.round(test_inv, decimals=12)

    # ================================================================================================================
    # Find the Drone to cam transformation:
    # ================================================================================================================
    # Find the rotation between camera and drone axis (The difference in (Drone rotation) and world rotation = 0,0,0)
    camera_rotation_matrix = cv_lib.get_3d_rotation_matrix_from_yaw_pitch_roll(
        yaw=camera_state_in_drone_coord.rotation.yaw,
        pitch=camera_state_in_drone_coord.rotation.pitch,
        roll=camera_state_in_drone_coord.rotation.roll,
        degrees=True)

    # Find the translation between camera coord to drone coord ( Since the location we have is in drone coord its easy)
    camera_translate_vector = np.array([camera_state_in_drone_coord.location.x,
                                        camera_state_in_drone_coord.location.y,
                                        camera_state_in_drone_coord.location.z])

    # Combine into transformation matrix from camera coord to drone coord
    camera_to_drone = cv_lib.find_transformation_matrix(camera_translate_vector, camera_rotation_matrix)

    # To find the inverse transformation we invert the matrix ( Hence now we have from drone to camera transformation)
    drone_to_camera = np.linalg.inv(camera_to_drone)


    def camera_to_drone(point):
        coord = camera_translate_vector + np.matmul(np.linalg.inv(camera_rotation_matrix), point)
        return np.round(coord, decimals=12)


    # ================================================================================================================
    # Check if true:
    # ================================================================================================================
    ship_translate_vector = np.array([ship_data.location.x, ship_data.location.y, ship_data.location.z, 1])
    ship_translate_vector = np.round(ship_translate_vector, decimals=12)

    ship_coord_in_world = np.reshape(ship_translate_vector, (4, 1))
    ship_coord_in_drone = np.matmul(world_to_drone, ship_coord_in_world)
    ship_coord_in_drone = np.round(ship_coord_in_drone, decimals=12)

    ship_coord_in_cam = np.matmul(drone_to_camera, ship_coord_in_drone)
    ship_coord_in_cam = np.round(ship_coord_in_cam, decimals=12)
    debug = 0

    # Altitude is in Meters
    # drone_altitude = 10
    # ship_altitude = 0  # 0 since this is unknown
    #
    # # GPS = array[latitude, longitude]
    # ship_gps_coord = np.array([10, 10, ship_altitude])
    # drone_gps_coord = np.array([10, 8, drone_altitude])
    # # Rotation between world and drone
    # rotation_global_to_drone = cv_lib.get_rotation_from_compass(compass_data=45.0)
    #
    # # Translation and Rotation between drone and camera
    # translation_drone_to_cam = np.array([0, 0, -0.2])
    # rotation_drone_to_cam = cv_lib.get_3d_rotation_matrix_from_yaw_pitch_roll(0, 90, 0)
