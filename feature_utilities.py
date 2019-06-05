import ast
import math
import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy import interpolate

__all__ = ["convert_types_in_dict", "moving_window_stride", "window_trapezoidal", "box_counting_dimension"]


def convert_types_in_dict(xml_dict):
    """
    Evaluates all dictionary entries into Python literal structure, as dictionary read from XML file is always string.
    If value can not be converted it passed as it is.
    :param xml_dict: Dict - Dictionary of XML entries
    :return: Dict - Dictionary with converted values
    """
    out = {}
    for el in xml_dict:
        try:
            out[el] = ast.literal_eval(xml_dict[el])
        except ValueError:
            out[el] = xml_dict[el]

    return out


def moving_window_stride(array, window, step):
    """
    Returns view of strided array for moving window calculation with given window size and step
    :param array: numpy.ndarray - input array
    :param window: int - window size
    :param step: int - step lenght
    :return: strided: numpy.ndarray - view of strided array, index: numpy.ndarray - array of indexes
    """
    stride = array.strides[0]
    win_count = math.floor((len(array) - window + step) / step)
    strided = as_strided(array, shape=(win_count, window), strides=(stride*step, stride))
    index = np.arange(window - 1, window + (win_count-1) * step, step)
    return strided, index


def window_trapezoidal(size, slope):
    """
    Return trapezoidal window of length size, with each slope occupying slope*100% of window
    :param size: int - window length
    :param slope: float - trapezoid parameter, each slope occupies slope*100% of window
    :return: numpy.ndarray - trapezoidal window
    """
    if slope > 0.5:
        slope = 0.5
    if slope == 0:
        return np.full(size, 1)
    else:
        return np.array([1 if ((slope * size <= i) & (i <= (1-slope) * size)) else (1/slope * i / size) if (i < slope * size) else (1/slope * (size - i) / size) for i in range(1, size + 1)])


def box_counting_dimension(sig, y_box_size_multiplier, subsampling):
    # Box-Counting Example:
    # https://gist.github.com/rougier/e5eafc276a4e54f516ed5559df4242c0#file-fractal-dimension-py-L25
    n = 2 ** np.floor(np.log(len(sig)) / np.log(2))
    n = int(np.log(n) / np.log(2))
    sizes = 2 ** np.arange(n, 1, -1)

    box_count = []
    for box_size in sizes:
        x_box_size = box_size
        y_box_size = box_size * y_box_size_multiplier

        sig_minimum = np.min(sig)

        box_occupation = np.zeros(
            [int(len(sig) / x_box_size) + 1, int((np.max(sig) - sig_minimum) / y_box_size) + 1])

        interp_func = interpolate.interp1d(np.arange(0, len(sig), 1), sig.reshape(1, len(sig))[0])
        x_interp = np.arange(0, len(sig) - 1 + 1 / subsampling, 1 / subsampling)
        sig_interp = interp_func(x_interp)

        for i in range(len(sig_interp)):
            x_box_id = int(x_interp[i] / x_box_size)
            y_box_id = int((sig_interp[i] - sig_minimum) / y_box_size)
            box_occupation[x_box_id, y_box_id] = 1

        box_count.append(np.sum(box_occupation))

    coefs = np.polyfit(np.log(1 / sizes), np.log(box_count), 1)
    return coefs[0]
