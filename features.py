import pandas as pd
import numpy as np
import utilities as ut
import math
import pyeeg
import xml.etree.ElementTree as ET
from scipy import optimize, stats
from numpy.lib.stride_tricks import as_strided


def calculate_feature(record, name, **kwargs):
    """
    Calculates feature given by params['name'] of given pandas.DataFrame. Feature is calculated for each Series with
    column name of "EMG_\d". Feature parameters are passed by Dict params, eg. {'name': RMS, 'window': 500, 'step': 250}
    :param record: pandas.DataFrame - input DataFrame with data to calculate features from
    :param name: string - name of the requested feature
    :param kwargs: parameters for feature calculation.
    :return: pandas.DataFrame - DataFrame containing output of desired feature
    """
    feature_func_name = 'feature_' + name  # Get feature function name based on name
    feature_values = pd.DataFrame()  # Create empty DataFrame

    for column in record.filter(regex="EMG_\d"):  # For each column containing EMG data (for each Series)
        feature_label = name + '_' + column.split('_')[1]  # Prepare feature column label
        feature_values[feature_label] = globals()[feature_func_name](record[column], **kwargs)  # Call feature calculation by function name, and add to output DataFtame

    return feature_values


def features_from_xml(xml_file_url, hdf5_file_url):
    """
    Calculates feature defined in given XML file containing feature names and parameters. See 'all_features.xml' for
    example. Calculate features of given putEMG record file in hdf5 format.
    :param xml_file_url: string - url to XML file containing feature descriptors
    :param hdf5_file_url: string - url to putEMG hdf5 record file
    :return: pandas.DataFrame - DataFrame containing output for all desired features
    """
    record = pd.read_hdf(hdf5_file_url)  # Read HDF5 file into pandas DataFrame
    feature_frame = pd.DataFrame()

    xml_root = ET.parse(xml_file_url).getroot()  # Load XML file with feature config
    for xml_entry in xml_root.iter('feature'):  # For each feature entry in XML file
        xml_entry.attrib = ut.convert_types_in_dict(xml_entry.attrib)  # Convert attribute dictrionary to Python literals
        feature_frame = feature_frame.join(calculate_feature(record, **xml_entry.attrib), how="outer")  # add to output frame values calculated by each feature function

    return feature_frame


def feature_IEMG(series, window, step):
    """Integrated EMG"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    return pd.Series(data=np.sum(np.abs(windows_strided), axis=1), index=series.index[indexes])


def feature_AAC(series, window, step):
    """Average Amplitude Change"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    # TODO: Check if really should by divided by N, not N-1 (standard mean), verification needed
    return pd.Series(data=np.divide(np.sum(np.abs(np.diff(windows_strided)), axis=1), window), index=series.index[indexes])


def feature_ApEn(series, window, step, m, r):
    """Approximate Entropy
    AnEn feature is using PyEEG library v0.4.0 as it is, licensed with GNU GPL v3
    http://pyeeg.org"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    return pd.Series(data=np.apply_along_axis(lambda win: pyeeg.ap_entropy(win, m, r), axis=1, arr=windows_strided), index=series.index[indexes])


def feature_AR(series, window, step, order, noisestd):
    """Auto-Regressive Coefficients"""
    # TODO: Verification needed, ARMA model coef calculated by fminsearch, min function not 100% sure, coefs output is Series of ndarrays, ok?
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)

    def AR_min_func(a, data, win):
        noise = np.random.normal(0, noisestd, size=stride_count)
        return np.sum(np.abs(windows_strided[win][order::order] - (np.sum(data * a, axis=1) + noise)))

    win_coefs = pd.Series()

    for widx in range(len(windows_strided)):
        stride = windows_strided[widx].strides[0]
        stride_count = math.floor((len(windows_strided[widx]) - 1) / order)
        win_strided = as_strided(windows_strided[widx], shape=[stride_count, order], strides=(stride * order, stride))

        a = optimize.fmin(func=AR_min_func, x0=np.full(order, 1), args=(win_strided, widx), disp=0)
        win_coefs.at[series.index[indexes[widx]]] = a

    return win_coefs


def feature_CC(series, window, step, order, noisestd):
    """Cepstral Coefficients"""
    # TODO: CC is derived from ARMA model coefficirnts, this approach requires minimizing AR model twice if both features are requested
    win_coefs = feature_AR(series, window, step, order, noisestd)
    for coefs in win_coefs:
        coefs[0] = -coefs[0]
        for p in range(1, order):
            coefs[p] = -coefs[p] - np.sum(
                [1 - (l / (p + 1)) for l in range(1, p + 1)] * np.full(p, coefs[p] * coefs[p - 1]))

    return win_coefs


def feature_DASDV(series, window, step):
    """Difference Absolute Standard Deviation Value"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    return pd.Series(data=np.sqrt(np.mean(np.square(np.diff(windows_strided)), axis=1)), index=series.index[indexes])


def feature_Kurt(series, window, step):
    """Kurtosis"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    return pd.Series(data=stats.kurtosis(windows_strided, axis=1), index=series.index[indexes])


def feature_LOG(series, window, step):
    """Log Detector"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    return pd.Series(data=np.exp(np.mean(np.log(np.abs(windows_strided)), axis=1)), index=series.index[indexes])


def feature_MAV1(series, window, step):
    """Modified Mean Absolute Value Type 1"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    win_weight = [1 if ((0.25*window <= i) & (i <= 0.75*window)) else 0.5 for i in range(1, window+1)]
    return pd.Series(data=np.mean(np.abs(windows_strided) * win_weight, axis=1), index=series.index[indexes])


def feature_MAV2(series, window, step):
    """Modified Mean Absolute Value Type 2"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    # TODO: Phinyomark states that window weight for i > 0.75N is 4(i-N)/4, should be 4(N-i)/N, verification needed
    # win_weight = [1 if ((0.25*window <= i) & (i <= 0.75*window)) else (4*i/window) if (i < 0.25*window) else (4*(window-i)/window) for i in range(1, window+1)]
    win_weight = ut.window_trapezoidal(window, 0.25)
    return pd.Series(data=np.mean(np.abs(windows_strided) * win_weight, axis=1), index=series.index[indexes])


def feature_MAV(series, window, step):
    """Mean Absolute Value"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    return pd.Series(data=np.mean(np.abs(windows_strided), axis=1), index=series.index[indexes])


def feature_MAVSLP(series, window, step):
    """Mean Absolute Value Slope"""
    # TODO: Mean Absolute Value Slope is considered as feature as set of k-segment diffs of adjectent MAV, here not the set/list but each diff is calculated, verification needed
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    # It can be also defined as below, however it not practical:
    # slp = np.diff(np.mean(np.abs(windows_strided), axis=1))
    # stride = slp.strides[0]
    # return pd.Series(data=as_strided(slp, shape=(len(slp) - segments + 1, segments), strides=(stride, stride)).tolist(), index=series.index[indexes[segments:]])
    return pd.Series(data=np.diff(np.mean(np.abs(windows_strided), axis=1)), index=series.index[indexes[1:]])


def feature_MHW(series, window, step):
    """Multiple Hamming Windows"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    return pd.Series(data=np.sum(np.square(windows_strided * np.hamming(window)), axis=1), index=series.index[indexes])


def feature_MTW(series, window, step, windowslope):
    """Multiple Trapezoidal Windows"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    return pd.Series(data=np.sum(np.square(windows_strided) * ut.window_trapezoidal(window, windowslope), axis=1), index=series.index[indexes])


def feature_MYOP(series, window, step, threshold):
    """Myopulse Percentage Rate"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    return pd.Series(data=np.sum(windows_strided > threshold, axis=1) / window, index=series.index[indexes])


def feature_RMS(series, window, step):
    """Root Mean Square"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    return pd.Series(data=np.sqrt(np.mean(np.square(windows_strided), axis=1)), index=series.index[indexes])


def feature_SampEn(series, window, step, m, r):
    """Sample Entropy
    SampEn feature is using PyEEG library v 0.02_r2 as it is, licensed with GNU GPL v3
    http://pyeeg.sourceforge.net/"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    return pd.Series(data=np.apply_along_axis(lambda win: pyeeg.samp_entropy(win, m, r), axis=1, arr=windows_strided), index=series.index[indexes])


def feature_Skew(series, window, step):
    """Skewness"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    return pd.Series(data=stats.skew(windows_strided, axis=1), index=series.index[indexes])


def feature_SSC(series, window, step, threshold):
    """Slope Sign Change"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    # TODO: reimplemented definition, needs verification
    return pd.Series(data=np.apply_along_axis(lambda x: np.sum(np.diff(np.diff(x[(x < -threshold) | (x > threshold)]) > 0)), axis=1, arr=windows_strided), index=series.index[indexes])


def feature_SSI(series, window, step):
    """Simple Square Integral"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    return pd.Series(data=np.sum(np.square(windows_strided), axis=1), index=series.index[indexes])


def feature_TM(series, window, step, order):
    """Absolute Temporal Moment"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    return pd.Series(data=np.abs(np.mean(np.power(windows_strided, order), axis=1)), index=series.index[indexes])


def feature_VAR(series, window, step):
    """Variance"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    return pd.Series(data=np.var(windows_strided, axis=1), index=series.index[indexes])


def feature_V(series, window, step, v):
    """V-Order"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    # TODO: Phinyomark sugests order of v=3 which will result in complex result, needed verification
    if v % 2 != 0:
        windows_strided = np.asarray(windows_strided, dtype=complex)
    return pd.Series(data=np.power(np.mean(np.power(windows_strided, v), axis=1), 1./v), index=series.index[indexes])


def feature_WAMP(series, window, step, threshold):
    """Willison Amplitude"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    return pd.Series(data=np.sum(np.diff(windows_strided) >= threshold, axis=1), index=series.index[indexes])


def feature_WL(series, window, step):
    """Waveform Length"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    return pd.Series(data=np.sum(np.diff(windows_strided), axis=1), index=series.index[indexes])


def feature_ZC(series, window, step, threshold):
    """Zero Crossing"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    zc = np.apply_along_axis(lambda x: np.sum(np.diff(x[(x < -threshold) | (x > threshold)] > 0)), axis=1, arr=windows_strided)
    return pd.Series(data=zc, index=series.index[indexes])


def feature_MNF(series, window, step):
    """Mean Frequency"""
    # TODO: In case of FD features FFT is calculated many times with XML approach, change approach?
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    power, freq = ut.power_spectrum(windows_strided, window)
    return pd.Series(data=np.sum(power*freq, axis=1) / np.sum(power, axis=1), index=series.index[indexes])


def feature_MDF(series, window, step):
    """Median Frequency"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    power, freq = ut.power_spectrum(windows_strided, window)
    TTPhalf = np.sum(power, axis=1)/2
    MDF = np.zeros(len(windows_strided))
    for w in range(len(power)):
        for s in range(1, len(power) + 1):
            if np.sum(power[w, :s]) > TTPhalf[w]:
                MDF[w] = freq[s - 1]
                break
    return pd.Series(data=MDF, index=series.index[indexes])


def feature_PKF(series, window, step):
    """Peak Frequency"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    power, freq = ut.power_spectrum(windows_strided, window)
    return pd.Series(data=freq[np.argmax(power, axis=1)], index=series.index[indexes])


def feature_MNP(series, window, step):
    """Mean Power"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    power, freq = ut.power_spectrum(windows_strided, window)
    return pd.Series(data=np.mean(power, axis=1), index=series.index[indexes])


def feature_TTP(series, window, step):
    """Total Power"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    power, freq = ut.power_spectrum(windows_strided, window)
    return pd.Series(data=np.sum(power, axis=1), index=series.index[indexes])


def feature_SM(series, window, step, order):
    """Spectral Moment"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    power, freq = ut.power_spectrum(windows_strided, window)
    return pd.Series(data=np.sum(np.power(power, order), axis=1), index=series.index[indexes])


def feature_FR(series, window, step, flb, fhb):
    """Frequency Ratio"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    power, freq = ut.power_spectrum(windows_strided, window)
    lb = np.sum(power[:, (flb[0] < freq) & (freq < flb[1])], axis=1)
    hb = np.sum(power[:, (fhb[0] < freq) & (freq < fhb[1])], axis=1)
    return pd.Series(data=(lb / hb), index=series.index[indexes])