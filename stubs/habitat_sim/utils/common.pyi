from __future__ import annotations
from _io import BytesIO
import _magnum
from habitat_sim._ext.habitat_sim_bindings.core import orthonormalize_rotation_shear
import magnum as mn
import math as math
import numpy
import numpy as np
import quaternion as qt
import quaternion
from urllib.request import urlopen
from zipfile import ZipFile
__all__: list[str] = ['BytesIO', 'ZipFile', 'angle_between_quats', 'colorize_ids', 'd3_40_colors_hex', 'd3_40_colors_rgb', 'download_and_unzip', 'math', 'mn', 'np', 'orthonormalize_rotation_shear', 'qt', 'quat_from_angle_axis', 'quat_from_coeffs', 'quat_from_magnum', 'quat_from_two_vectors', 'quat_rotate_vector', 'quat_to_angle_axis', 'quat_to_coeffs', 'quat_to_magnum', 'random_quaternion', 'urlopen']
def angle_between_quats(q1: _magnum.Quaternion, q2: _magnum.Quaternion) -> float:
    """
    Computes the angular distance between two magnum quaternions
    
        :return: The angular distance between q1 and q2 in radians
        
    """
def colorize_ids(ids):
    ...
def download_and_unzip(file_url, local_directory):
    ...
def quat_from_angle_axis(theta: float, axis: numpy.ndarray) -> quaternion.quaternion:
    """
    Creates a quaternion from angle axis format
    
        :param theta: The angle to rotate about the axis by
        :param axis: The axis to rotate about
        :return: The quaternion
        
    """
def quat_from_coeffs(coeffs: typing.Union[typing.Sequence[float], numpy.ndarray]) -> quaternion.quaternion:
    """
    Creates a quaternion from the coeffs returned by the simulator backend
    
        :param coeffs: Coefficients of a quaternion in :py:`[b, c, d, a]` format,
            where :math:`q = a + bi + cj + dk`
        :return: A quaternion from the coeffs
        
    """
def quat_from_magnum(quat: _magnum.Quaternion) -> quaternion.quaternion:
    ...
def quat_from_two_vectors(v0: numpy.ndarray, v1: numpy.ndarray) -> quaternion.quaternion:
    """
    Creates a quaternion that rotates the first vector onto the second vector
    
        :param v0: The starting vector, does not need to be a unit vector
        :param v1: The end vector, does not need to be a unit vector
        :return: The quaternion
    
        Calculates the quaternion q such that
    
        .. code:: py
    
            v1 = quat_rotate_vector(q, v0)
        
    """
def quat_rotate_vector(q: quaternion.quaternion, v: numpy.ndarray) -> numpy.ndarray:
    """
    Helper function to rotate a vector by a quaternion
    
        :param q: The quaternion to rotate the vector with
        :param v: The vector to rotate
        :return: The rotated vector
    
        Does
    
        .. code:: py
    
            v = (q * qt.quaternion(0, *v) * q.inverse()).imag
        
    """
def quat_to_angle_axis(quat: quaternion.quaternion) -> typing.Tuple[float, numpy.ndarray]:
    """
    Converts a quaternion to angle axis format
    
        :param quat: The quaternion
        :return:
            -   `float` --- The angle to rotate about the axis by
            -   `numpy.ndarray` --- The axis to rotate about. If :math:`\\theta = 0`,
                then this is hardcoded to be the +x axis
        
    """
def quat_to_coeffs(quat: quaternion.quaternion) -> numpy.ndarray:
    """
    Converts a quaternion into the coeffs format the backend expects
    
        :param quat: The quaternion
        :return: Coefficients of a quaternion in :py:`[b, c, d, a]` format,
            where :math:`q = a + bi + cj + dk`
        
    """
def quat_to_magnum(quat: quaternion.quaternion) -> _magnum.Quaternion:
    ...
def random_quaternion():
    """
    Convenience function to sample a random Magnum::Quaternion.
        See http://planning.cs.uiuc.edu/node198.html.
        
    """
d3_40_colors_hex: list = ['0x1f77b4', '0xaec7e8', '0xff7f0e', '0xffbb78', '0x2ca02c', '0x98df8a', '0xd62728', '0xff9896', '0x9467bd', '0xc5b0d5', '0x8c564b', '0xc49c94', '0xe377c2', '0xf7b6d2', '0x7f7f7f', '0xc7c7c7', '0xbcbd22', '0xdbdb8d', '0x17becf', '0x9edae5', '0x393b79', '0x5254a3', '0x6b6ecf', '0x9c9ede', '0x637939', '0x8ca252', '0xb5cf6b', '0xcedb9c', '0x8c6d31', '0xbd9e39', '0xe7ba52', '0xe7cb94', '0x843c39', '0xad494a', '0xd6616b', '0xe7969c', '0x7b4173', '0xa55194', '0xce6dbd', '0xde9ed6']
d3_40_colors_rgb: numpy.ndarray  # value = array([[ 31, 119, 180],...
