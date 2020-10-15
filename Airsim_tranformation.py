import numpy as np

from Data.UE4Data import State
from Data.camera_info import CameraInfo
from Transformations.transform import TRotation, Transform
from base_tranformation import BaseTransformation


class AirSimTransformation(BaseTransformation):
    # Static data:
    def __init__(self,
                 show_segment_points,
                 number_of_segment_points,
                 camera_info: CameraInfo,
                 staticShipData: dict,
                 airsim_gimble_bug: bool
                 ):
        # Convert to airsim scale:
        for key in staticShipData.keys():
            staticShipData[key] = staticShipData[key] / 100
        super().__init__(show_segment_points, number_of_segment_points, camera_info, staticShipData)
        self.airsim_gimble_bug = airsim_gimble_bug

    def get_drone_to_camera_transformation(self, gimbal: bool, drone_data: State, world_to_drone: Transform):
        # ================================================================================================================
        # Find drone to camera transform_matrix
        # ================================================================================================================
        # Since the camera information is static atm we just use static values:
        rotation = self.camera_state_in_drone_coord.rotation
        location = self.camera_state_in_drone_coord.location
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
            if self.airsim_gimble_bug:  # Check if weird bug is active (Locking all axis)
                r2 = world_to_drone.inverse().get_rotation()
            else:
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

    def perspective_projection(self, location: np.ndarray):
        """
         NOTE UNREAL x = Depth, y = Width, z = Height,
         so we start converting these for easy understanding
         :returns Film Coords x=width y=height (x=0 , y=0 is center)
        """
        X = location[1]
        Y = location[2]
        Z = location[0]

        x = self.camera_info.focal * X / Z
        y = self.camera_info.focal * Y / Z
        return x, y

    def pixel_coord(self, x: float, y: float, height: int, width: int):
        """procentage"""
        ratio = width / height
        u = (x + 1) / 2  # * width
        v = (y * ratio + 1) / 2  # * height
        return u, v

    def transform_to_frame(self, ship_to_camera, point):
        point_camera = ship_to_camera(point)
        x, y = self.perspective_projection(point_camera)
        u, v = self.pixel_coord(x, y, self.camera_info.image_height, self.camera_info.image_width)
        return u[0], v[0]

    def getShipSegmentsLocationInImageFrame(self, drone_data: State, ship_data: State):
        assert drone_data is not None
        assert ship_data is not None
        # ================================================================================================================
        # Find world to Drone transform_matrix
        # ================================================================================================================
        world_to_drone, drone_to_world = self.get_transformation_and_inv(drone_data.location, drone_data.rotation,
                                                                         degree=True)
        # ================================================================================================================
        # Find world to Ship transform_matrix
        # ================================================================================================================
        world_to_ship, ship_to_world = self.get_transformation_and_inv(ship_data.location, ship_data.rotation,
                                                                       degree=True)
        # ================================================================================================================
        # Find Drone to Camera transform_matrix (Static since we know that this is never going to change)
        # ================================================================================================================
        # NOTE UNREAL x = Depth, z = Height, y = Width
        drone_to_camera, camera_to_drone = self.get_drone_to_camera_transformation(gimbal=True,
                                                                                   drone_data=drone_data,
                                                                                   world_to_drone=world_to_drone)

        # ================================================================================================================
        # Compute ship points in world frame into drone frame into camera frame into screen space
        # ================================================================================================================
        # Combine transformation
        def ship_to_camera(ship_point_local):
            point_world = ship_to_world(ship_point_local)
            point_drone = world_to_drone(point_world)
            point_camera = drone_to_camera(point_drone)
            return point_camera

        # convert segment_points
        for ship_side in self.ship_sides.values():
            for segment in ship_side.list_of_segments:
                segment.p_from_I = self.transform_to_frame(ship_to_camera, segment.p_from_S)
                segment.p_to_I = self.transform_to_frame(ship_to_camera, segment.p_to_S)
            ship_side.ship_corner_from.p_corner_I = self.transform_to_frame(ship_to_camera,
                                                                            ship_side.ship_corner_from.p_corner_S)
            ship_side.ship_corner_to.p_corner_I = self.transform_to_frame(ship_to_camera,
                                                                          ship_side.ship_corner_to.p_corner_S)
            # p_from = self.transform_to_frame(ship_to_camera, ship_segment.p_from)
            # p_to = self.transform_to_frame(ship_to_camera, ship_segment.p_to)
            # frame_segments.append(ShipSegment(p_from, p_to, ship_segment.ship_side, ship_segment.seg_nr))
        ship_point = self.transform_to_frame(ship_to_camera, np.array([0, 0, 0]).reshape((3, 1)))

        return self.ship_sides, ship_point
