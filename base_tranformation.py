from abc import ABC, abstractmethod

import numpy as np

from Data.UE4Data import Rotation, State, ShipSide
from Data.camera_info import CameraInfo
from Transformations.cv_lib import np_swap_values
from Transformations.transform import Transform, TRotation


class BaseTransformation(ABC):

    def __init__(self,
                 show_segment_points,
                 number_of_segment_points,
                 camera_info: CameraInfo,
                 staticShipData: dict):
        self.show_segment_points = show_segment_points
        self.number_of_segment_points = number_of_segment_points
        self.camera_info = camera_info
        self.should_run = False
        self.ship_model_points_l = self.compute_ship_model_points(staticShipData)
        # BGR
        self.color_map = {'fc': (255, 0, 0),
                          'fr': (255, 128, 0),
                          'fl': (255, 255, 0),
                          'br': (0, 255, 0),
                          'bl': (0, 0, 255),
                          'seg_point': (0, 225, 225)
                          }
        self.ship_sides = self.compute_ship_sides_and_sub_segments(self.ship_model_points_l)
        self.camera_state_in_drone_coord = camera_info.camera_state_in_drone_coord

    @staticmethod
    def find_front_offset(ship_length):
        return -ship_length * 0.07
        # if 0 < ship_length < 100:
        #     return - (ship_length * 0.07)
        # if ship_length < 150:
        #     return -(ship_length * 0.10)
        # if ship_length < 200:
        #     return -(ship_length * 0.13)
        # return -(ship_length * 0.16)

    def compute_ship_model_points(self,
                                  staticShipData,
                                  swap: bool = False):
        """
        :param staticShipData:
        :param swap: False= face along x+, True = Face along y+
        :return: a dictionary with points
        """

        dist_to_bow = staticShipData["dist_gps_to_front"]
        dist_to_stern = staticShipData["dist_gps_to_back"]
        dist_to_starboard = staticShipData["dist_gps_to_starboard"]
        dist_to_port = staticShipData["dist_gps_to_port"]
        dist_from_deck_to_water = staticShipData["dist_gps_to_water_level"]

        # TODO: Check when not centered
        ship_length = dist_to_bow + dist_to_stern
        center_offset = dist_to_starboard - dist_to_port
        p_front_center = np.array([dist_to_bow, 0, 0])
        p_back_center = np.array([-dist_to_stern, 0, 0])
        p_right_center = np.array([0, dist_to_starboard, 0])
        p_left_center = np.array([0, -dist_to_port, 0])

        front_offset = np.array([self.find_front_offset(ship_length), 0, 0])

        p_back_left_corner = p_back_center + p_left_center
        p_back_right_corner = p_back_center + p_right_center
        p_front_left_corner = p_front_center + front_offset + p_left_center
        p_front_right_corner = p_front_center + front_offset + p_right_center

        # TODO: Could be wrong when GPS is not centered. Test when real data is here.
        p_front_center[1] = center_offset
        p_back_center[1] = center_offset
        if swap:
            p_back_left_corner = np_swap_values(p_back_left_corner, 0, 1)
            p_back_right_corner = np_swap_values(p_back_right_corner, 0, 1)
            p_front_left_corner = np_swap_values(p_front_left_corner, 0, 1)
            p_front_right_corner = np_swap_values(p_front_right_corner, 0, 1)
            p_front_center = np_swap_values(p_front_center, 0, 1)

        p_back_left_corner[-1] = -dist_from_deck_to_water
        p_back_right_corner[-1] = -dist_from_deck_to_water
        p_front_left_corner[-1] = -dist_from_deck_to_water
        p_front_right_corner[-1] = -dist_from_deck_to_water
        p_front_center[-1] = -dist_from_deck_to_water

        result = {
            'fc': p_front_center,
            'fr': p_front_right_corner,
            'br': p_back_right_corner,
            'bl': p_back_left_corner,
            'fl': p_front_left_corner,
        }
        return result

    @staticmethod
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

    def get_drone_to_camera_transformation(self, gimbal: bool, drone_data: State, world_to_drone: Transform):
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
            r2 = self.rotation_matrix_from_vectors(drone_z, world_z)
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

    def get_transformation_and_inv(self, location, rotation, degree: bool, order='xyz',
                                   is_left_handed=None, translate_before_rotate=True) -> (Transform, Transform):

        # Find the rotation between the two coordinate system
        rotation = self.createRotation(rotation, degree, is_left_handed, order)

        # Find the translation between the two coordinate system
        location_world = np.array([location.x, location.y, location.z]).reshape(3, 1)
        world_to_location_translation = -location_world

        # world_to_drone
        world_to_location = Transform(rotation=rotation,
                                      translation=world_to_location_translation,
                                      translate_before_rotate=translate_before_rotate)
        # drone_to_world
        location_to_world = world_to_location.inverse()
        return world_to_location, location_to_world

    @staticmethod
    def createRotation(rotation: Rotation, degree, is_left_handed=None, order='xyz') -> TRotation:
        """
        rotation_x = TRotation(rotation.roll, axis='x', degrees=degree, left_hand=True)
        rotation_y = TRotation(rotation.pitch, axis='y', degrees=degree, left_hand=True)
        rotation_z = TRotation(rotation.yaw, axis='z', degrees=degree, left_hand=True)
        rotation_xy = rotation_x.after(rotation_y)
        rotation_xyz = rotation_xy.after(rotation_z)
        :param rotation:
        :param degree:
        :param is_left_handed:
        :param order:
        :return:
        """
        assert len(order) == 3
        assert 'x' in order
        assert 'y' in order
        assert 'z' in order
        if is_left_handed is None:
            is_left_handed = [True, True, True]
        last_rotation = TRotation(rotation.getRotationAroundAxis(axis=order[0]), axis=order[0], degrees=degree,
                                  left_hand=is_left_handed[0])
        mid_rotation = TRotation(rotation.getRotationAroundAxis(axis=order[1]), axis=order[1], degrees=degree,
                                 left_hand=is_left_handed[1])
        first_rotation = TRotation(rotation.getRotationAroundAxis(axis=order[2]), axis=order[2], degrees=degree,
                                   left_hand=is_left_handed[2])
        rot = last_rotation.after(mid_rotation)
        rot = rot.after(first_rotation)
        return rot

    def compute_ship_sides_and_sub_segments(self, ship_model_points_l: dict) -> dict:
        # Clockwise generate the side of the ship
        back = ShipSide("back", ship_model_points_l["br"], ship_model_points_l["bl"])
        left = ShipSide("left", ship_model_points_l["bl"], ship_model_points_l["fl"])
        f_left = ShipSide("f_left", ship_model_points_l["fl"], ship_model_points_l["fc"])
        f_right = ShipSide("f_right", ship_model_points_l["fc"], ship_model_points_l["fr"])
        right = ShipSide("f_left", ship_model_points_l["fr"], ship_model_points_l["br"])

        # Pack data into dict
        result = {"back": back,
                  "left": left,
                  "f_left": f_left,
                  "f_right": f_right,
                  "right": right}

        # Distributed the sub segments based on len of ship sides
        segments_to_spend = self.number_of_segment_points
        while segments_to_spend > 0:
            # Find biggest seg and add point
            max_side = None
            max_len = 0
            for side_name in result.keys():
                side = result[side_name]
                if side.get_dist_per_segment() > max_len:
                    max_side = side
                    max_len = side.get_dist_per_segment()
            max_side.sub_segments += 1
            segments_to_spend -= 1

        # Generate the sub segments for each side
        for side_name in result.keys():
            result[side_name].generate_sub_segments()

        return result

    @abstractmethod
    def getShipSegmentsLocationInImageFrame(self,
                                            drone_data: State,
                                            ship_data: State):
        pass

    @abstractmethod
    def perspective_projection(self, location: np.ndarray):
        pass

    @abstractmethod
    def pixel_coord(self, x: float, y: float, height: int, width: int):
        pass
