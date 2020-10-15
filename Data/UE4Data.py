import math
from random import randint

import airsim
import cv2
import numpy as np

from Transformations import cv_lib


class Point2D:
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y

    def as_tuple(self):
        return self.x, self.y

    def __str__(self):
        return str(self.as_tuple())

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if not isinstance(other, Point2D):
            return False
        if self.x != other.x:
            return False
        if self.y != other.y:
            return False
        return True

    def __ne__(self, other):
        return not self == other

    def from_np(self, array):
        self.x = array[0]
        self.y = array[1]
        return self

    def from_tuple(self, t):
        self.x = t[0]
        self.y = t[1]
        return self


class Location:
    def __init__(self, location_dict=None, x=0.0, y=0.0, z=0.0):
        if location_dict is not None:
            self.x = location_dict['x']
            self.y = location_dict['y']
            self.z = location_dict['z']

        else:
            self.x = x
            self.y = y
            self.z = z

    def __str__(self):
        return "x: " + str(self.x) + " y: " + str(self.y) + " z: " + str(self.z)

    def as_np_array(self):
        return np.array([self.x, self.y, self.z], dtype=np.float).reshape(3, 1)

    def to_dict(self):
        location_dict = {'x': self.x, 'y': self.y, 'z': self.z}
        return location_dict


class Rotation:
    def __init__(self, rotation_dict):
        self.pitch = rotation_dict['pitch']
        self.roll = rotation_dict['roll']
        self.yaw = rotation_dict['yaw']
        self.d = rotation_dict

    def __str__(self):
        return "pitch:" + str(self.pitch) + " roll: " + str(self.roll) + " yaw. " + str(self.yaw)

    def as_np_array(self):
        return np.array([self.pitch, self.roll, self.yaw]).reshape(3, 1)

    def to_dict(self):
        rotation_dict = {'pitch': self.pitch, 'roll': self.roll, 'yaw': self.yaw}
        return rotation_dict

    def getRotationAroundAxis(self, axis: str):
        if axis is 'x':
            return self.roll
        elif axis is 'y':
            return self.pitch
        else:
            return self.yaw


class State:
    def __init__(self, location, rotation):
        self.location = Location(location)
        self.rotation = Rotation(rotation)

    def __str__(self):
        return str(self.location) + "\n" + str(self.rotation)

    def to_dict(self):
        d1 = self.location.to_dict()
        d2 = self.rotation.to_dict()
        d3 = {
            'location': d1,
            'rotation': d2
        }
        return d3

    @staticmethod
    def from_dict(d):
        return State(location=d['location'], rotation=d['rotation'])

    @property
    def x(self):
        return self.location.x

    @property
    def y(self):
        return self.location.y

    @property
    def z(self):
        return self.location.z

    @property
    def yaw(self):
        return self.rotation.yaw

    @property
    def pitch(self):
        return self.rotation.pitch

    @property
    def roll(self):
        return self.rotation.roll


# Wrapper
class Drone(State):
    def __init__(self, location, rotation, world_to_local=None, local_to_world=None):
        super().__init__(location, rotation)
        self.world_to_local = world_to_local
        self.local_to_world = local_to_world


# Wrapper
class Ship(State):
    def __init__(self, location, rotation):
        super().__init__(location, rotation)


def convert_data(data: dict) -> (Drone, Ship):
    drone = None
    ship = None
    for v_type in data.keys():
        location = data[v_type][0]
        rotation = data[v_type][1]
        if v_type == 'Drone':
            drone = Drone(location, rotation)
        elif v_type == 'Ship':
            ship = Ship(location, rotation)
    return drone, ship


def convert_airsim_data(data: airsim.Pose):
    pitch, roll, yaw = airsim.to_eularian_angles(data.orientation)

    rotation = {
        "pitch": cv_lib.radians_to_degree(pitch),
        "roll": cv_lib.radians_to_degree(roll),
        "yaw": cv_lib.radians_to_degree(yaw)
    }
    location = {
        "x": data.position.x_val,
        "y": data.position.y_val,
        "z": data.position.z_val
    }
    state = State(location, rotation)
    return state


# bgr
color_test = {
    "front left": (0, 0, 0),
    "front right": (0, 255, 0),
    "right": (0, 0, 255),
    "back": (255, 0, 0),
    "left": (255, 255, 0)
}


class ShipSide:
    def __init__(self, name, corner_from_W, corner_to_W):
        self.name = name
        self.corner_from_S = corner_from_W
        self.corner_to_S = corner_to_W

        # Every side takes care of one corner and this corner is the start corner
        self.ship_corner_from = ShipCorner(self.corner_from_S)
        self.ship_corner_to = ShipCorner(self.corner_to_S)  # Need this to check if point belongs to next section
        self.list_of_segments = []

        # Used to decide how many sub segments is needed for this side
        self.side_dist = np.linalg.norm(corner_to_W - corner_from_W)
        self.sub_segments = 1

    def __str__(self):
        return self.name

    def get_dist_per_segment(self):
        return self.side_dist / self.sub_segments

    def generate_sub_segments(self):
        lastly = self.corner_from_S
        for i in range(self.sub_segments):
            t = (i + 1) * (1 / self.sub_segments)
            point = (1 - t) * self.corner_from_S + t * self.corner_to_S
            ss = ShipSegment(lastly, point)
            self.list_of_segments.append(ss)
            lastly = point

    def shortest_dist_to_point(self, img, point, w, h, nr_of_sub_segments=20):
        min_p = self.ship_corner_from.to_pixel(w, h)
        min_d = self.ship_corner_from.find_min_dist(img, point, w, h)
        min_seg = self.ship_corner_from
        for seg in self.list_of_segments:
            changed, keep_going, min_dist, min_point = seg.find_min_dist(img, point,
                                                                         w, h,
                                                                         nr_of_sub_segments,
                                                                         min_d, min_p)
            min_d = min_dist
            min_p = min_point
            if changed:
                min_seg = seg
            if not keep_going:
                break
        return min_seg, min_p, min_d

    def T_shortest_dist_to_point(self, img, point, w, h):
        p_to_check = Point2D().from_tuple(point)
        p_from = Point2D().from_tuple(self.ship_corner_from.to_pixel(w, h))
        p_to = Point2D().from_tuple(self.ship_corner_to.to_pixel(w, h))
        p_on_line = Point2D().from_tuple(self.p4(p_from.as_tuple(), p_to.as_tuple(), p_to_check.as_tuple()))
        a, b = self.slope_intercept(p_from.as_tuple(), p_to.as_tuple())

        tmp = img.copy()
        h_n, w_n = tmp.shape[:2]
        show_line(tmp, (0, b), (w_n, a * w_n + b), (w_n, h_n), (w, h), (255, 0, 0))
        show_line(tmp, p_to_check.as_tuple(), p_on_line.as_tuple(), (w_n, h_n), (w, h), (0, 255, 0))
        show_img(tmp)

        min_p = p_from
        min_d = self.ship_corner_from.find_min_dist(img, p_to_check.as_tuple(), w, h)
        min_seg = self.ship_corner_from

        # Check if point on line is exactly one of the corners (Special case)
        if p_on_line == p_to or p_on_line == p_from:
            return min_seg, min_p.as_tuple(), min_d

        for seg in self.list_of_segments:
            tmp2 = tmp.copy()
            seg.draw(tmp2)
            if seg.contain_point(p_to_check, w, h):
                min_seg = seg
                min_p = p_on_line
                min_d = squared_distance(p_on_line.as_tuple(), p_to_check.as_tuple())
                do_text(tmp2, (20, 20), (0, 0), "Its in the segment")
                show_img(tmp2)
                break
            do_text(tmp2, (20, 20), (0, 0), "Its not in the segment")
            show_img(tmp2)
        return min_seg, min_p.as_tuple(), min_d

    def draw_side(self, img):
        # Draw corner if valid:
        self.ship_corner_from.draw(img)
        for seg in self.list_of_segments:
            seg.draw(img)

    def reset(self):
        self.ship_corner_from.reset()
        for seg in self.list_of_segments:
            seg.reset()

    @staticmethod
    def slope_intercept(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        a = (y2 - y1) / (x2 - x1)
        b = y1 - a * x1
        return a, b

    @staticmethod
    def p4(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        dx, dy = x2 - x1, y2 - y1
        det = dx * dx + dy * dy
        a = (dy * (y3 - y1) + dx * (x3 - x1)) / det
        return x1 + a * dx, y1 + a * dy


class ShipCorner:
    def __init__(self, p_corner_S):
        self.p_corner_S = p_corner_S
        self.p_corner_I = None
        self.shortest_dist = np.inf
        self.short_dist_p_from = None
        self.short_dist_p_to = None

        self.color = (randint(0, 255), randint(0, 255), randint(0, 255))

    def __str__(self):
        return "SC <" + str(self.p_corner_S) + ">"

    def __repr__(self):
        return str(self)

    def to_pixel(self, w, h):
        return self.point_to_pixel(w, h, self.p_corner_I)

    @staticmethod
    def point_to_pixel(w, h, point):
        return np.array([int(point[0] * w), int(point[1] * h)])

    def find_min_dist(self, img, p1, w, h):
        return squared_distance(self.to_pixel(w, h), p1)

    def reset(self):
        self.p_corner_I = None
        self.shortest_dist = np.inf
        self.short_dist_p_from = None
        self.short_dist_p_to = None

    def update_dist(self, min_dist, min_point_from, min_point_to, w, h):
        if min_dist < self.shortest_dist:
            self.shortest_dist = min_dist
            self.short_dist_p_from = (min_point_from[0] / w, min_point_from[1] / h)
            self.short_dist_p_to = (min_point_to[0] / w, min_point_to[1] / h)

    def draw(self, img):
        h, w = img.shape[:2]
        if self.shortest_dist < np.inf:
            p_from = self.point_to_pixel(w, h, self.short_dist_p_from)
            p_to = self.point_to_pixel(w, h, self.short_dist_p_to)
            cv2.line(img, (p_from[0], p_from[1]), (p_to[0], p_to[1]), color=self.color, thickness=2)
        center = self.to_pixel(w, h)
        cv2.circle(img, center=(center[0], center[1]), radius=7, color=self.color, thickness=-1)


class ShipSegment:
    def __init__(self, p_from, p_to):
        # Local ship coordinates (Static)
        self.p_from_S = p_from
        self.p_to_S = p_to
        self.color = (randint(0, 255), randint(0, 255), randint(0, 255))

        # Image frame coordinates (Dynamic and will change):
        self.p_from_I = None
        self.p_to_I = None
        self.shortest_dist = np.inf
        self.short_dist_p_from = None
        self.short_dist_p_to = None

    def __str__(self):
        return "SS <" + str(self.p_from_S) + "> -> <" + str(self.p_to_S) + ">"

    def __repr__(self):
        return str(self)

    def to_pixel(self, w, h):
        return self.point_to_pixel(w, h, self.p_from_I), self.point_to_pixel(w, h, self.p_to_I)

    @staticmethod
    def point_to_pixel(w, h, point):
        return np.array([int(point[0] * w), int(point[1] * h)])

    def contain_point(self, point: Point2D, w, h):
        epsilon = 0.00001
        p_from = Point2D(self.p_from_I[0] * w, self.p_from_I[1] * h)
        p_to = Point2D(self.p_to_I[0] * w, self.p_to_I[1] * h)

        crossproduct = (point.y - p_from.y) * (p_to.x - p_from.x) - (point.x - p_from.x) * (p_to.y - p_from.y)

        # compare versus epsilon for floating point values, or != 0 if using integers
        if abs(crossproduct) > epsilon:
            return False

        dotproduct = (point.x - p_from.x) * (p_to.x - p_from.x) + (point.y - p_from.y) * (p_to.y - p_from.y)
        if dotproduct < 0:
            return False

        squaredlengthba = (p_to.x - p_from.x) * (p_to.x - p_from.x) + (p_to.y - p_from.y) * (p_to.y - p_from.y)
        if dotproduct > squaredlengthba:
            return False

        return True

    def find_min_dist(self, img, p1, w, h, nr_of_sub_segments, current_min_d, current_min_p):
        p_from, p_to = self.to_pixel(w, h)
        debug = False
        tmp = None
        n_h, n_w = None, None
        if debug:
            tmp = img.copy()
            n_h, n_w = tmp.shape[:2]
            show_line(tmp, current_min_p, p1, (n_w, n_h), (w, h), (255, 0, 0), current_min_d,
                      (0, 0))
            dist_to = squared_distance(p_to, p1)
            show_line(tmp, p_to, p1, (n_w, n_h), (w, h), (0, 255, 0), dist_to,
                      (0, 0))

        changed = False
        keep_going = True
        min_dist = current_min_d
        min_point = current_min_p
        in_seg, in_first_half, d, p = self.check_if_point_in_segment(min_dist, p_from, p_to, p1,
                                                                     nr_of_sub_segments)
        if not in_seg:
            if debug:
                pass
                # show_line(tmp, p, p1, d,
                #           (0, 0), (n_w, n_h), (w, h),(0,0,255))
                # show_img(tmp)
            changed = True
            return changed, keep_going, d, p
        if debug:
            show_img(tmp)
        keep_going = False
        # Binary search
        start = 0 if in_first_half else math.floor(nr_of_sub_segments / 2) - 1
        end = math.ceil(nr_of_sub_segments / 2) + 1 if in_first_half else nr_of_sub_segments - 1
        while start <= end:
            mid = (start + end) // 2
            dist, p2 = self.check_dist(mid, p1, p_from, p_to, nr_of_sub_segments)

            if dist >= min_dist:
                end = mid - 1
            if dist < min_dist:
                changed = True
                min_dist = dist
                min_point = p2
                start = mid + 1
            if debug:
                tmp2 = tmp.copy()
                show_line(tmp2, p2, p1, (n_w, n_h), (w, h), (0, 0, 150), dist,
                          (0, 50))
                show_img(tmp2)
        # for i in range(nr_of_sub_segments - 1):
        #     t = (i + 1) * (1 / nr_of_sub_segments)
        #     p2 = (1 - t) * p_from + t * p_to
        #     dist = squared_distance(p1, p2)
        #     if dist > min_dist:
        #         keep_going = False
        #         break
        #     changed = True
        #     min_dist = dist
        #     min_point = p2
        return changed, keep_going, min_dist, min_point

    @staticmethod
    def check_dist(index, point, p_from, p_to, nr_of_sub_segments):
        t = (index + 1) * (1 / nr_of_sub_segments)
        p2 = (1 - t) * p_from + t * p_to
        dist = squared_distance(point, p2)
        return dist, p2

    def reset(self):
        self.p_from_I = None
        self.p_to_I = None
        self.shortest_dist = np.inf
        self.short_dist_p_from = None
        self.short_dist_p_to = None

    def update_dist(self, min_dist, min_point_from, min_point_to, w, h):
        if min_dist < self.shortest_dist:
            self.shortest_dist = min_dist
            self.short_dist_p_from = (min_point_from[0] / w, min_point_from[1] / h)
            self.short_dist_p_to = (min_point_to[0] / w, min_point_to[1] / h)

    def draw(self, img):
        h, w = img.shape[:2]
        # If distance found
        if self.shortest_dist < np.inf:
            p_from = self.point_to_pixel(w, h, self.short_dist_p_from)
            p_to = self.point_to_pixel(w, h, self.short_dist_p_to)
            cv2.line(img, (p_from[0], p_from[1]), (p_to[0], p_to[1]), color=self.color, thickness=2)
        # Draw segment
        p_from, p_to = self.to_pixel(w, h)
        cv2.line(img, (p_from[0], p_from[1]), (p_to[0], p_to[1]), self.color, thickness=3)

    @staticmethod
    def check_if_point_in_segment(curr_min_dist, p_from, p_to, p1, nr_of_sub_segments):
        """
        :param curr_min_dist:
        :param p_from: start of segment
        :param p_to: end of segment
        :param p1: point on contour
        :param nr_of_sub_segments:
        :return: is_in_seg:bool, is_in_first_half:bool, min_dist:float, p_right_before_to:Tuple
        """
        dist_to = squared_distance(p_to, p1)
        if curr_min_dist <= dist_to:
            # Since distance from is shorter than dist to, it must be in the segment
            return True, True, None, None
        t = ((nr_of_sub_segments - 1) / nr_of_sub_segments)
        p_right_before_to = (1 - t) * p_from + t * p_to
        dist3 = squared_distance(p_right_before_to, p1)
        if dist3 <= dist_to:
            # Since distance from is shorter than dist to, it must be in the segment
            return True, False, None, None
        return False, False, dist3, p_right_before_to


def squared_distance(p1, p2):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2


def show_step(image, point_on_c, p_from, p_to, p):
    cv2.circle(image, point_on_c, radius=3, color=(0, 255, 0), thickness=3)
    cv2.circle(image, p_from, radius=3, color=(0, 0, 255), thickness=3)
    cv2.circle(image, p_to, radius=3, color=(0, 0, 255), thickness=3)
    cv2.circle(image, p, radius=2, color=(0, 255, 255), thickness=3)


def show_line(tmp, p_from, p_to, n_s: tuple, s: tuple, color: tuple, dist: float = 0., offset=(0, 0), show_dots=False,
              show_dist=False):
    p_from_scaled = convert_size(p_from, n_s[0], n_s[1], s[0], s[1])
    p_to_scaled = convert_size(p_to, n_s[0], n_s[1], s[0], s[1])
    if show_dots:
        cv2.circle(tmp, center=p_from_scaled, radius=5, color=(0, 255, 0), thickness=-1)
        cv2.circle(tmp, center=p_to_scaled, radius=5, color=(255, 0, 0),
                   thickness=-1)
    cv2.line(tmp, p_from_scaled, p_to_scaled, color=color, thickness=2)
    if show_dist:
        do_text(tmp, p_from_scaled, offset, dist)


def do_text(image, p, offset, text):
    pos = (int(p[0]) + int(offset[0]), int(p[1]) + int(offset[1]))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 0, 255)
    line_type = 2
    cv2.putText(image, str(text), pos, font, font_scale, font_color, line_type)


def show_img(img):
    cv2.imshow("image", img)
    if cv2.waitKey(0):
        cv2.destroyAllWindows()


def convert_size(p, w, h, old_w, old_h):
    return int(p[0] * w / old_w), int(p[1] * h / old_h)
