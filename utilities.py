import ast
import math
import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy import signal


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


def power_spectrum(windows_strided, winsize):
    """
    Calculates power spectrum of given windowed signal
    :param windows_strided: numpy.ndarray - 2D numpy array containing windowed signal
    :param winsize: int - window size
    :return: power: numpy.ndarray - 2D array with power spectrum, freq: numpy.ndarray - array with frequency vector
    """
    T = 1.0 / 5120
    power = 2.0 / winsize * np.absolute(np.fft.fft(windows_strided)[:, :winsize//2])
    freq = np.fft.fftfreq(winsize, d=T)[:winsize//2]
    #power[:, ((freq <= 10) | (freq >= 500))] = 0
    #return signal.medfilt(power, 3), freq
    return power, freq