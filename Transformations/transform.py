from copy import deepcopy as copy
import numpy as np

from Transformations import conversions as conv
from scipy.spatial.transform import Rotation as R


###################################################
# CLASSES                                         #
###################################################
class Rotation(object):
    def __init__(self, angle=0, axis='x', degrees=True, left_hand=False):
        """Set Rotation by angle about axis.

            Parameters
            ----------
            angle : float, optional
                (rad) rotation angle. Default: 0
            axis : Vector or int, optional
                Axis about which the rotation is performed. Can be a Vector is
                arbitraty direction or an integer indication one of the
                coordinate axes. Default: 0 (i. e. x-axis)
        """
        if degrees:
            angle = np.radians(angle)
        self.axis_order = 'zyx'
        self.matrix = None
        self.axis_list = ['x', 'y', 'z']
        self.set_angle_and_axis(angle, axis, left_hand)

    def __repr__(self):
        return 'Rotation(axis=%s) with matrix\n%s' % (repr(self.axis_order), repr(self.matrix))

    def __mul__(self, other):
        return self.dot(other)

    def __truediv__(self, other):
        if isinstance(other, Rotation):
            return self.__mul__(other.inverse())
        else:
            raise TypeError()

    def __eq__(self, other):
        assert isinstance(other, Rotation)
        Meq = self.matrix == other.matrix
        return np.sum(Meq) == 9

    def __call__(self, v):
        return self.apply_to(v)

    ###################################################
    # UNARY OPERATORS                                 #
    ###################################################
    def shape(self):
        return self.matrix.shape

    shape = property(shape)

    def inverse(self):
        """
        Return inverse Rotation.
        By transposing the rotation matrix and reverse the axis_order

        """
        return Rotation().set_matrix(self.matrix.T, self.axis_order[::-1])

    def rotate(self, angle, axis):
        """Additionally rotate the Rotation.

            Parameters
            ----------
            angle : float
                (rad) rotation angle
            axis : int or char
                Axis about which the rotation is performed. Can be a Vector or
                the number of the coordinate axis (0:x, 1:y, 2:z).

            Returns
            -------
            Rotation
        """
        other = Rotation(angle, axis)
        new = other.after(self)
        self.set_matrix(new.get_matrix(), new.axis_order)
        return self

    ###################################################
    # BINARY OPERATORS                                #
    ###################################################
    def dot(self, other):
        """Return result of dot-product with Rotation or np dnarray"""
        if isinstance(other, Rotation):
            A = self.get_matrix()
            B = other.get_matrix()
            matrix = A.dot(B)
            axis = self.axis_order + other.axis_order
            return Rotation().set_matrix(matrix, axis)
        elif isinstance(other, np.ndarray):
            M = self.get_matrix()
            v = other
            # Check if compatible
            if v.shape[0] != M.shape[0]:
                v = v.reshape((M.shape[0], 1))
            return M.dot(v)
        elif isinstance(other, Scaling):
            A = self.get_matrix()
            B = other.get_matrix()
            matrix = A.dot(B)
            return matrix
        else:
            raise TypeError('other must be Rotation or np ndarray.')

    def after(self, other):
        """Return combined Rotation.
            test_r = x_rotation.dot(y_rotation).dot(z_rotation)
            p = np.dot(z_rotation, point)
            p = np.dot(y_rotation, p)
            p = np.dot(x_rotation, p)
            p2 = test_r.dot(point)
            p2 == test_r = True
        """
        return self.dot(other)

    def before(self, other):
        """Return combined Rotation."""
        return other.after(self)

    ###################################################
    # SETTERS                                         #
    ###################################################
    def set_matrix(self, matrix, axis):
        """Set rotation matrix."""
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (3, 3)
        self.matrix = matrix
        self.axis_order = axis
        return self

    def set_angle_and_axis(self, angle: float, axis, left_hand):
        """Set Rotation by angle and axis.
            :param angle: float - (rad) rotation angle
            :param axis: int or char - Axis about which the rotation is performed. Can be a char (x,y,z)
                                       or the number of the coordinate axis (0:x, 1:y, 2:z).
            :param left_hand: bool - decides the rotation direction (left = clockwise, right = anti-clockwise)
        """

        # special case: no rotation
        if angle == 0:
            self.matrix = np.eye(3)
            return self

        # special case: coordinate axis
        if isinstance(axis, int) and axis in range(3):
            # delegate to sub-function
            axis = self.axis_list[axis]
            return self.set_angle_and_axis(angle, axis, left_hand)

        # input check
        if axis not in self.axis_list:
            raise TypeError('`axis` must be an int between 0..2 or char x,y,z.')

        self.axis_order = axis
        # shortcuts
        sin = np.sin(angle)
        cos = np.cos(angle)

        # M[row][col]
        if axis == 'x':
            x_rotation = np.eye(3, dtype=np.float)
            x_rotation[1][1] = cos
            x_rotation[1][2] = sin if left_hand else -sin
            x_rotation[2][2] = cos
            x_rotation[2][1] = -sin if left_hand else sin
            self.matrix = x_rotation
        elif axis == 'y':
            y_rotation = np.eye(3, dtype=np.float)
            y_rotation[0][0] = cos
            y_rotation[0][2] = -sin if left_hand else sin
            y_rotation[2][0] = sin if left_hand else -sin
            y_rotation[2][2] = cos
            self.matrix = y_rotation
        elif axis == 'z':
            z_rotation = np.eye(3, dtype=np.float)
            z_rotation[0][0] = cos
            z_rotation[0][1] = sin if left_hand else -sin
            z_rotation[1][0] = -sin if left_hand else sin
            z_rotation[1][1] = cos
            self.matrix = z_rotation

        return self

    ###################################################
    # GETTERS                                         #
    ###################################################
    def get_matrix(self):
        """Return rotation matrix."""
        return self.matrix

    def pad(self, axis_order):
        if len(axis_order) == 3:
            return axis_order
        aol = list(axis_order)
        for item in self.axis_list:
            if item not in aol:
                axis_order = axis_order + str(item)
        return axis_order

    def get_angle(self):
        """Return rotation angle (in degrees)."""
        r = R.from_matrix(self.matrix.T)
        # ao = self.pad(self.axis_order)
        angles = r.as_euler('xyz', degrees=True)
        result = {}
        for axis in list(self.axis_order):
            i = self.axis_list.index(axis)
            result[axis] = angles[i]
        return result

    def get_axis_order(self):
        """
            Return rotation axis_order
            it is read from right to left
            example:
            >> xyz = first z then y then x
            >> xyz * point ==
                point = z.dot(point)
                point = y.dot(point)
                point = x.dot(point)
        """
        return self.axis_order

    ###################################################
    # APPLY                                           #
    ###################################################
    def apply_to(self, v: np.ndarray):
        """Return rotated Vector."""
        assert isinstance(v, np.ndarray)
        return self.dot(v)


class Scaling(object):
    def __init__(self, x=1, y=1, z=1, n=3):
        """
        :param x:
        :param y:
        :param z:
        :param n: min is 3
        """
        m_scaling = np.identity(n=n)  # (n,n)
        m_scaling[0][0] = x
        m_scaling[1][1] = y
        m_scaling[2][2] = z
        self.matrix = m_scaling

    def __repr__(self):
        return 'Scaling with matrix\n%s' % (repr(self.matrix))

    def __mul__(self, other):
        return self.matrix.dot(other)

    def __truediv__(self, other):
        if isinstance(other, Scaling):
            return self.__mul__(np.linalg.inv(other.matrix))
        else:
            raise TypeError()

    def __eq__(self, other):
        assert isinstance(other, Scaling)
        Meq = self.matrix == other.matrix
        return np.sum(Meq) == 9

    def __call__(self, v: np.ndarray):
        return self.apply_to(v)

    def shape(self):
        return self.matrix.shape

    shape = property(shape)

    def get_matrix(self):
        return self.matrix

    def apply_to(self, v: np.ndarray):
        M = self.matrix
        return M.dot(v)


class Transform(object):
    def __init__(self,
                 translation: np.array = np.zeros((3, 1)),
                 rotation: Rotation = Rotation(),
                 scaling: Scaling = Scaling(),
                 translate_before_rotate=False):

        """Initialize.

            Parameters
            ----------
            translation : Vector, optional
                Default: null-vector
            rotation : Rotation, optional
                Default: identity rotation
            translate_before_rotate : bool, optional
                If True, `shift` is applied before `rotation`, otherwise inverse.
                Default: False.
        """
        self.scaling = scaling
        self.rotation = rotation

        # Have to be before translation
        if not translate_before_rotate:
            self.translation = translation
        else:
            self.translation = self.rotate(translation)

    def __repr__(self):
        """Return string representation."""
        t = repr(self.translation)
        s = repr(self.scaling)
        R = repr(self.rotation_as_euler())
        return 'Transformation with scale by\n%s\n after translate by\n%s\n lastly rotation by\n%s' % (s, t, R)

    def __call__(self, v, decimals=12):
        return self.apply_to(v, decimals)

    ###################################################
    # UNARY OPERATOR                                  #
    ###################################################
    def inverse(self):
        """Return inverse Transform."""
        t = self.translation
        r = self.rotation
        s = self.scaling
        return Transform(-t, r.inverse(), s, translate_before_rotate=True)

    ###################################################
    # GETTERS                                         #
    ###################################################
    def get_translation(self):
        """Return translation Vector."""
        return self.translation

    def get_rotation(self):
        """Return Rotation object."""
        return self.rotation

    def get_scaling(self):
        return self.scaling

    def apply_to(self, v: np.ndarray, decimals: int = 12) -> np.ndarray:
        """Return transformed Vector."""
        t = self.translation
        if v.shape != t.shape:
            v = v.reshape(t.shape)
        v = self.scale(v)
        v = self.rotate(v)
        v = self.translate(v)
        return np.round(v, decimals=decimals)

    def scale(self, v: np.ndarray) -> np.ndarray:
        if v.shape[0] != self.scaling.shape[0]:
            v = v.reshape((self.scaling.shape[0], 1))
        return self.scaling.apply_to(v)

    def rotate(self, v: np.ndarray) -> np.ndarray:
        if v.shape[0] != self.rotation.shape[0]:
            v = v.reshape((self.rotation.shape[0], 1))
        return self.rotation.apply_to(v)

    def translate(self, v: np.ndarray) -> np.ndarray:
        if v.shape != self.translation.shape:
            v = v.reshape(self.translation.shape)
        return v + self.translation

    def to_matrix(self):
        r = self.rotation
        t = self.translation
        m = np.hstack((r, t))
        m = np.vstack((m, [0, 0, 0, 1]))
        return m

    def rotation_as_euler(self, order='zyx'):
        r = R.from_matrix(self.rotation.get_matrix().T)
        return r.as_euler(order, degrees=True)


###################################################
# TESTING                                         #
###################################################
# The functions have all been tested.
#
if __name__ == '__main__':
    pass
