import io
import operator
import time

import airsim
import cv2
import numpy as np
from PIL import Image

from Data.UE4Data import State, convert_airsim_data
from Transformations.cv_lib import np_swap_values
from Transformations.transform import Transform, TRotation

camera_focal_length = 1
camera_fov = 90

# Unreal coordinates
# camera_state_in_drone_coord = State({'x': 0.0, 'y': 0.0, 'z': -5.0},
#                                     {'pitch': -90, 'roll': -180, 'yaw': -90})

# Airsim coordinate
camera_state_in_drone_coord = State({'x': 0.05, 'y': 0.0, 'z': .0},
                                    {'pitch': 270, 'roll': 0, 'yaw': 0})
# BGR
color_map = {'fc': (255, 0, 0),
             'fr': (255, 128, 0),
             'fl': (255, 255, 0),
             'br': (0, 255, 0),
             'bl': (0, 0, 255),
             'seg_point': (0, 225, 225)
             }

# Airsim has this weird bug with only be able to lock all axis or none. So if this is set, we lock all axis.
airsim_gimble_bug = True


def rotation_matrix_from_vectors(vec1, vec2, left_hand=True):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)

    v = np.cross(a, b)

    c = np.dot(a, b)
    s = np.linalg.norm(v)  # If s is 0 then a = b meaning no rotation needed
    # Check if any rotation is needed
    if s == 0.0:
        return TRotation(axis='xyz')
    if left_hand:
        kmat = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]])
    else:
        kmat = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return TRotation(axis='xyz').set_matrix(rotation_matrix)


def get_drone_to_camera_transformation(gimbal: bool, drone_data: State, world_to_drone: Transform):
    # ================================================================================================================
    # Find drone to camera transform_matrix
    # ================================================================================================================
    # Since the camera information is static atm we just use static values:
    global camera_state_in_drone_coord
    rotation = camera_state_in_drone_coord.rotation
    location = camera_state_in_drone_coord.location
    # Find the rotation between camera and drone axis
    camera_rotation_x = TRotation(rotation.roll, axis='x', degrees=True, left_hand=True)
    camera_rotation_y = TRotation(rotation.pitch, axis='y', degrees=True, left_hand=True)
    camera_rotation_z = TRotation(rotation.yaw, axis='z', degrees=True, left_hand=True)
    camera_rotation_xy = camera_rotation_x.after(camera_rotation_y)
    camera_rotation_xyz = camera_rotation_xy.after(camera_rotation_z)

    if gimbal:
        # We want to align the z axis of world and drone:
        # TODO: https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
        # Find world and drone z vector:
        world_z = np.array([0, 0, 1]).reshape((3, 1))
        r = world_to_drone.get_rotation()
        drone_z = r.apply_to(world_z)
        # Calculate the rotation.
        if airsim_gimble_bug:  # Check if weird bug is active (Locking all axis)
            r2 = world_to_drone.inverse().get_rotation()
        else:
            r2 = rotation_matrix_from_vectors(drone_z, world_z)
        camera_rotation_xyz = camera_rotation_xyz.after(r2)

    # Find the translation between camera coord to drone coord
    camera_pos_d = np.array([location.x, location.y, location.z]).reshape(3, 1)
    drone_to_camera_trans = -camera_pos_d

    # Invert the translation
    drone_to_camera = Transform(rotation=camera_rotation_xyz,
                                translation=drone_to_camera_trans,
                                translate_before_rotate=False)
    camera_to_drone = drone_to_camera.inverse()

    return drone_to_camera, camera_to_drone


def find_front_offset(ship_length):
    return -ship_length * 0.07
    # if 0 < ship_length < 100:
    #     return - (ship_length * 0.07)
    # if ship_length < 150:
    #     return -(ship_length * 0.10)
    # if ship_length < 200:
    #     return -(ship_length * 0.13)
    # return -(ship_length * 0.16)


def compute_ship_model_points(dist_to_bow: float,
                              dist_to_stern: float,
                              dist_to_starboard: float,
                              dist_to_port: float,
                              dist_from_water_to_deck: float,
                              swap: bool = False):
    """
    :param dist_to_bow: distance from gps antenna to bow(front) (meters)
    :param dist_to_stern: distance from gps antenna to stern(back) (meters)
    :param dist_to_starboard: distance from gps antenna to starboard(right) (meters)
    :param dist_to_port: distance from gps antenna to port(left) (meters)
    :param swap: False= face along x+, True = Face along y+
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

    if swap:
        p_back_left_corner = np_swap_values(p_back_left_corner, 0, 1)
        p_back_right_corner = np_swap_values(p_back_right_corner, 0, 1)
        p_front_left_corner = np_swap_values(p_front_left_corner, 0, 1)
        p_front_right_corner = np_swap_values(p_front_right_corner, 0, 1)
        p_front_center = np_swap_values(p_front_center, 0, 1)

    p_back_left_corner[-1] = dist_from_water_to_deck
    p_back_right_corner[-1] = dist_from_water_to_deck
    p_front_left_corner[-1] = dist_from_water_to_deck
    p_front_right_corner[-1] = dist_from_water_to_deck
    p_front_center[-1] = dist_from_water_to_deck

    result = {
        'fc': p_front_center,
        'fr': p_front_right_corner,
        'br': p_back_right_corner,
        'bl': p_back_left_corner,
        'fl': p_front_left_corner,
    }
    return result


def getImage(client: airsim.MultirotorClient) -> np.array:
    """
    Airsim returns a series of bytes, which we convert to a PIL image and then into a RGBA numpy array.
    The RGBA image have the alpha channel always be 255, we wish to
    @param camera_name:
    @return:
    """
    img_bytes = client.simGetImage(camera_name="down_center_custom", image_type=airsim.ImageType.Scene)
    if len(img_bytes) < 10:
        return getImage(client)
    rgba_image = Image.open(io.BytesIO(img_bytes))
    rgb_image = rgba_image.convert('RGB')
    np_image = np.copy(np.asarray(rgb_image))
    rgb_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
    np_image = np.copy(np.asarray(rgb_image))
    return np_image


def get_transformation_and_inv(location, rotation, degree: bool) -> (Transform, Transform):
    # Find the rotation between drone and world axis (The difference in (Drone rotation) and world rotation = 0,0,0)
    rotation_x = TRotation(rotation.roll, axis='x', degrees=degree, left_hand=True)
    rotation_y = TRotation(rotation.pitch, axis='y', degrees=degree, left_hand=True)
    rotation_z = TRotation(rotation.yaw, axis='z', degrees=degree, left_hand=True)
    rotation_xy = rotation_x.after(rotation_y)
    rotation_xyz = rotation_xy.after(rotation_z)

    # Find the translation between drone coord to world coord ( Since the location we have is in world coord its easy)
    location_world = np.array([location.x, location.y, location.z]).reshape(3, 1)
    world_to_location_translation = -location_world

    # world_to_drone
    world_to_location = Transform(rotation=rotation_xyz,
                                  translation=world_to_location_translation,
                                  translate_before_rotate=True)
    # drone_to_world
    location_to_world = world_to_location.inverse()
    return world_to_location, location_to_world


def perspective_projection(location: np.ndarray, focal: float):
    """
     NOTE UNREAL x = Depth, y = Width, z = Height,
     so we start converting these for easy understanding
     :returns Film Coords x=width y=height (x=0 , y=0 is center)
    """
    X = location[1]
    Y = location[2]
    Z = location[0]

    x = focal * X / Z
    y = focal * Y / Z
    return x, y


def pixel_coord(x: float, y: float, height: int, width: int):
    ratio = width / height
    u = (x + 1) / 2 * width
    v = (y * ratio + 1) / 2 * height
    return u, v


def render_segment_points(ship_model_points_l: dict, number_of_segment_points: int) -> list:
    # unpacking
    points = list(ship_model_points_l.values())
    if number_of_segment_points < len(points):
        number_of_segment_points = len(points)

    # Create Pairs and compute total dist:
    dist = 0
    list_of_outer_seg = {}
    seg_curr_dist = {}
    seg_dist = {}
    number_of_point_pr_seg = {}
    lastly = points[-1]
    for p in points:
        pair = (lastly, p)  # pair is each point making up a side of the ship
        curr_dist = np.linalg.norm(p - lastly)
        dist += curr_dist
        list_of_outer_seg[str(pair)] = pair
        number_of_point_pr_seg[str(pair)] = 0  # Always add one point in each side of the ship
        seg_dist[str(pair)] = curr_dist
        seg_curr_dist[str(pair)] = curr_dist
        lastly = p

    # Distributed the rest of the points
    segments_to_spend = number_of_segment_points - len(number_of_point_pr_seg)
    while segments_to_spend > 0:
        # Find biggest seg and add point
        pair_str = max(seg_curr_dist.items(), key=operator.itemgetter(1))[0]
        number_of_point_pr_seg[pair_str] += 1
        seg_curr_dist[pair_str] = seg_dist[pair_str] / (number_of_point_pr_seg[pair_str] + 1)
        segments_to_spend -= 1

    result = []
    for pair_str in number_of_point_pr_seg.keys():
        nr_point = number_of_point_pr_seg[pair_str]
        p_from, p_to = list_of_outer_seg[pair_str]
        result.append(p_from)
        for i in range(nr_point):
            t = (i + 1) * (1 / (nr_point + 1))
            point = (1 - t) * p_from + t * p_to
            result.append(point)
    return result


if __name__ == '__main__':
    show_segment_points = True
    number_of_segment_points = 20

    # ================================================================================================================
    # Connect to UE4 Server
    # ================================================================================================================
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)

    # ================================================================================================================
    # Compute Static ship data
    # ================================================================================================================
    staticShipData = [10840 / 100, 11615 / 100, 1550 / 100, 1630 / 100,
                      -770 / 100]  # [front,back,left,right, height]
    ship_model_points_l = compute_ship_model_points(staticShipData[0],
                                                    staticShipData[1],
                                                    staticShipData[3],
                                                    staticShipData[2],
                                                    staticShipData[4])

    segment_point = render_segment_points(ship_model_points_l, number_of_segment_points) if show_segment_points else []
    # ================================================================================================================
    # Extract UE4 Data (LEFT-HAND axis :: Rotations = roll: Right-handed, pitch: Right-handed, yaw: Left-handed)
    # ================================================================================================================
    while True:
        client.simPause(True)
        np_rgb_image = getImage(client)
        drone_data = convert_airsim_data(client.simGetVehiclePose())
        camera_data = convert_airsim_data(client.simGetObjectPose("BP_PIPCamera_C_5"))
        ship_data = convert_airsim_data(client.simGetObjectPose("BP_Container_ship2_2"))
        test_data = convert_airsim_data(client.simGetObjectPose("test_3"))
        t1 = client.simGetGroundTruthKinematics()
        t2 = client.simGetGroundTruthEnvironment()
        client.simPause(False)

        assert drone_data is not None
        assert ship_data is not None

        # Unreal coordinates
        # staticShipData = [108, 116, 16, 16]  # [front,back,left,right]
        # Airsim coordinates

        drone_location = drone_data.location
        drone_rotation = drone_data.rotation
        ship_location = ship_data.location
        ship_rotation = ship_data.rotation
        # print("Camera:", str(camera_data))
        # print("Drone:", str(drone_data))
        # print("sub:\n", camera_data.location.as_np_array() - drone_data.location.as_np_array())
        # ================================================================================================================
        # Find world to Drone transform_matrix
        # ================================================================================================================
        world_to_drone, drone_to_world = get_transformation_and_inv(drone_location, drone_rotation, degree=True)
        # ================================================================================================================
        # Find world to Ship transform_matrix
        # ================================================================================================================
        world_to_ship, ship_to_world = get_transformation_and_inv(ship_location, ship_rotation, degree=True)
        # ================================================================================================================
        # Find world to Camera
        # ================================================================================================================
        world_to_camera, camera_to_world = get_transformation_and_inv(camera_data.location, camera_data.rotation, degree=True)

        # ================================================================================================================
        # Find Drone to Camera transform_matrix (Static since we know that this is never going to change)
        # ================================================================================================================
        # NOTE UNREAL x = Depth, z = Height, y = Width
        # cam_loc_in_drone = world_to_drone(camera_data.location.as_np_array())
        # camera_state_in_drone_coord.location.x = cam_loc_in_drone[0][0]
        # camera_state_in_drone_coord.location.y = cam_loc_in_drone[1][0]
        # camera_state_in_drone_coord.location.z = cam_loc_in_drone[2][0]
        drone_to_camera, camera_to_drone = get_drone_to_camera_transformation(gimbal=True,
                                                                              drone_data=drone_data,
                                                                              world_to_drone=world_to_drone)

        # ================================================================================================================
        # Project into image plane (http://www.cse.psu.edu/~rtc12/CSE486/lecture12.pdf)
        # ================================================================================================================
        # MOVED OUT OF WHILE
        # def perspective_projection(location: np.ndarray, focal: float):
        #     """
        #      NOTE UNREAL x = Depth, y = Width, z = Height,
        #      so we start converting these for easy understanding
        #      :returns Film Coords x=width y=height (x=0 , y=0 is center)
        #     """
        #     X = location[1]
        #     Y = location[2]
        #     Z = location[0]
        #
        #     x = focal * X / Z
        #     y = focal * Y / Z
        #     return x, y
        #
        #
        # def pixel_coord(x: float, y: float, height: int, width: int):
        #     ratio = width / height
        #     u = (x + 1) / 2 * width
        #     v = (y * ratio + 1) / 2 * height
        #     return u, v

        # ================================================================================================================
        # Compute ship points in world frame into drone frame into camera frame into screen space
        # ================================================================================================================
        height, width = np_rgb_image.shape[:2]


        # Mid of ship
        # ship_point_world = np.array([ship_location.x, ship_location.y, ship_location.z]).reshape(3, 1)
        # ship_point_drone = world_to_drone(ship_point_world)
        # ship_point_camera = drone_to_camera(ship_point_drone)
        # x, y = perspective_projection(ship_point_camera, camera_focal_length)
        # u, v = pixel_coord(x, y, height, width)

        # cv2.circle(np_rgb_image, center=(u, v), radius=10, color=(255, 128, 255), thickness=-1)

        # Test:
        # test_point_world = np.array([test_data.x, test_data.y, test_data.z]).reshape(3, 1)
        # test_point_drone = world_to_drone(test_point_world)
        # test_point_camera = drone_to_camera(test_point_drone)
        # x, y = perspective_projection(test_point_camera, camera_focal_length)
        # u, v = pixel_coord(x, y, height, width)
        # cv2.circle(np_rgb_image, center=(u, v), radius=10, color=(255, 128, 255), thickness=-1)

        # Combine transformation
        def ship_to_camera(ship_point_local):
            point_world = ship_to_world(ship_point_local)
            point_drone = world_to_drone(point_world)
            point_camera = drone_to_camera(point_drone)
            # point_camera = world_to_camera(point_world)
            x, y = perspective_projection(point_camera, camera_focal_length)
            u, v = pixel_coord(x, y, height, width)
            return u, v


        # Render key points:
        for key in ship_model_points_l:
            color = color_map[key]
            ship_point_local = ship_model_points_l[key].reshape(3, 1)
            u, v = ship_to_camera(ship_point_local)
            cv2.circle(np_rgb_image, center=(u, v), radius=10, color=color, thickness=-1)

        # Render segment_points
        result = []

        for point in segment_point:
            color = color_map["seg_point"]
            u, v = ship_to_camera(point)
            cv2.circle(np_rgb_image, center=(u, v), radius=4, color=color, thickness=-1)
            result.append((u, v))

        # draw lines:
        lastly = result[-1]
        for camera_point in result:
            cv2.line(np_rgb_image, lastly, camera_point, color_map["seg_point"])
            lastly = camera_point

        # start_c = np.array([0, 0, 0]).reshape(3,1)
        # start_d = camera_to_drone(start_c)
        # start_w = drone_to_world(start_d)
        #
        # stop_c = np.array([0.5, 0, 0]).reshape(3,1)
        # stop_d = camera_to_drone(stop_c)
        # stop_w = drone_to_world(stop_d)
        #
        # q = airsim.to_quaternion(pitch=0, roll=0, yaw=0)
        # start_pose = airsim.Pose(airsim.Vector3r(start_w[0][0], start_w[1][0], start_w[2][0]), q)
        # stop_pose = airsim.Pose(airsim.Vector3r(stop_w[0][0], stop_w[1][0], stop_w[2][0]), q)
        # # print(start_pose)
        # # print(stop_pose)
        # client.simSetObjectPose("arrow_start", start_pose)
        # client.simSetObjectPose("arrow_stop", stop_pose)
        cv2.imshow("image", np_rgb_image)

        debug = 0
        if cv2.waitKey(1) == 27:
            break  # esc to quit
        time.sleep(1)
    cv2.destroyAllWindows()
    # client.reset()
