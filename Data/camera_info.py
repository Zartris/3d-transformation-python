from Data.UE4Data import State


class CameraInfo:
    def __init__(self, focal: float, FOV: float,
                 image_width: int, image_height: int,
                 camera_state_in_drone_coord: State,
                 sensor_width: float = 1., sensor_height: float = 1.):
        """
        :param focal: in mm
        :param FOV: in degrees
        :param image_width: in pixels
        :param image_height: in pixels
        :param sensor_width: in mm
        :param sensor_height: in mm
        """
        self.focal = focal
        self.FOV = FOV
        self.image_width = image_width
        self.image_height = image_height
        self.sensor_width = sensor_width
        self.sensor_height = sensor_height
        self.camera_state_in_drone_coord = camera_state_in_drone_coord

    def getGSD(self, height):
        """
        :param height: Distance above water in meters
        :return: pixel to cm conversion
        """
        GSD_w = (height * self.sensor_width) / (self.focal * self.image_width)  # In cm
        GSD_h = (height * self.sensor_height) / (self.focal * self.image_height)  # In cm
        return GSD_h, GSD_w
