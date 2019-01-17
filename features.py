import pandas as pd
import numpy as np
import utilities as ut
import pyeeg
import xml.etree.ElementTree as ET
from scipy import stats, signal
from numpy.lib.stride_tricks import as_strided
import time


def calculate_feature(record: pd.DataFrame, name, **kwargs):
    """
    Calculates feature given name of given pandas.DataFrame. Feature is calculated for each Series with
    column name of "EMG_\d". Feature parameters are passed by **kwargs, eg. window=500, step=250
    :param record: pandas.DataFrame - input DataFrame with data to calculate features from
    :param name: string - name of the requested feature
    :param kwargs: parameters for feature calculation.
    :return: pandas.DataFrame - DataFrame containing output of desired feature
    """
    feature_func_name = 'feature_' + name  # Get feature function name based on name
    feature_values = pd.DataFrame()  # Create empty DataFrame

    start = time.time()
    print('Calculating feature ' + name + ':', end='', flush=True)
    for column in record.filter(regex="EMG_\d+"):  # For each column containing EMG data (for each Series)
        print(' ' + column.split('_')[1], end='', flush=True)
        feature_label = name + '_' + column.split('_')[1]  # Prepare feature column label
        # Call feature calculation by function name, and add to output DataFrame
        feature = globals()[feature_func_name](record[column], **kwargs)
        if isinstance(feature, pd.Series):
            feature_values[feature_label] = feature
        elif isinstance(feature, pd.DataFrame):
            d = {}
            for c in feature.columns:
                d[c] = feature_label + "_" + c
            feature = feature.rename(columns=d)
            feature_values = feature_values.join(feature, how='outer')

    print('', flush=True)
    print("Elapsed time: {:.2f}s".format(time.time() - start))

    return feature_values


def features_from_xml(xml_file_url, hdf5_file_url):
    """
    Calculates feature defined in given XML file containing feature names and parameters. See 'all_features.xml' for
    example. Calculate features of given putEMG record file in hdf5 format.
    :param xml_file_url: string - url to XML file containing feature descriptors
    :param hdf5_file_url: string - url to putEMG hdf5 record file
    :return: pandas.DataFrame - DataFrame containing output for all desired features
    """
    record: pd.DataFrame = pd.read_hdf(hdf5_file_url)  # Read HDF5 file into pandas DataFrame
    feature_frame = pd.DataFrame()

    xml_root = ET.parse(xml_file_url).getroot()  # Load XML file with feature config
    for xml_entry in xml_root.iter('feature'):  # For each feature entry in XML file
        # Convert attribute dictionary to Python literals
        xml_entry.attrib = ut.convert_types_in_dict(xml_entry.attrib)
        # add to output frame values calculated by each feature function
        feature_frame = feature_frame.join(calculate_feature(record, **xml_entry.attrib), how="outer")

    for other_data in list(record.filter(regex="^(?!EMG_).*")):
        feature_frame[other_data] = record.loc[feature_frame.index, other_data]

    return feature_frame


def feature_IAV(series, window, step):
    """Integral Absolute Value"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    return pd.Series(data=np.sum(np.abs(windows_strided), axis=1), index=series.index[indexes])


def feature_AAC(series, window, step):
    """Average Amplitude Change"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    return pd.Series(data=np.divide(np.sum(np.abs(np.diff(windows_strided)), axis=1), window),
                     index=series.index[indexes])


def feature_ApEn(series, window, step, m, r):
    """Approximate Entropy
    AnEn feature is using PyEEG library v0.4.0 as it is, licensed with GNU GPL v3
    http://pyeeg.org"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    return pd.Series(data=np.apply_along_axis(lambda win: pyeeg.ap_entropy(win, m, r),
                                              axis=1, arr=windows_strided), index=series.index[indexes])


def feature_AR(series, window, step, order) -> pd.DataFrame:
    """Auto-Regressive Coefficients"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)

    column_names = [str(i) for i in range(0, order)]
    win_coefs = pd.DataFrame(index=series.index[indexes], columns=column_names, dtype=np.float64)

    for widx in range(len(windows_strided)):
        stride = windows_strided[widx].strides[0]
        stride_count = len(windows_strided[widx]) - order
        x = as_strided(windows_strided[widx], shape=[stride_count, order], strides=(stride, stride))
        y = windows_strided[widx][order:]

        a, _, _, _ = np.linalg.lstsq(x, y, rcond=None)

        win_coefs.loc[series.index[indexes[widx]], :] = a
    return win_coefs


def feature_CC(series, window, step, order):
    """Cepstral Coefficients"""
    win_coefs = feature_AR(series, window, step, order)
    coefs = win_coefs.values
    coefs[:, 0] = -coefs[:, 0]
    for r in range(0, coefs.shape[0]):
        for p in range(1, order):
            coefs[r, p] = -coefs[r, p] - np.sum(
                [1 - (l / (p + 1)) for l in range(1, p + 1)] * np.full(p, coefs[r, p] * coefs[r, p - 1]))
    win_coefs.loc[:, :] = coefs
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
    win_weight = ut.window_trapezoidal(window, 0.25)
    return pd.Series(data=np.mean(np.abs(windows_strided) * win_weight, axis=1), index=series.index[indexes])


def feature_MAV(series, window, step):
    """Mean Absolute Value"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    return pd.Series(data=np.mean(np.abs(windows_strided), axis=1), index=series.index[indexes])


def feature_MAVSLP(series, window, step):
    """Mean Absolute Value Slope"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    return pd.Series(data=np.diff(np.mean(np.abs(windows_strided), axis=1)), index=series.index[indexes[1:]])


def feature_MHW(series, window, step):
    """Multiple Hamming Windows"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    return pd.Series(data=np.sum(np.square(windows_strided * np.hamming(window)), axis=1), index=series.index[indexes])


def feature_MTW(series, window, step, windowslope):
    """Multiple Trapezoidal Windows"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    return pd.Series(data=np.sum(np.square(windows_strided) * ut.window_trapezoidal(window, windowslope), axis=1),
                     index=series.index[indexes])


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
    return pd.Series(data=np.apply_along_axis(lambda win: pyeeg.samp_entropy(win, m, r), axis=1, arr=windows_strided),
                     index=series.index[indexes])


def feature_Skew(series, window, step):
    """Skewness"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    return pd.Series(data=stats.skew(windows_strided, axis=1), index=series.index[indexes])


def feature_SSC(series, window, step, threshold):
    """Slope Sign Change"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    return pd.Series(data=np.apply_along_axis(lambda x: np.sum((np.diff(x[:-1]) * np.diff(x[1:])) <= -threshold),
                                              axis=1, arr=windows_strided), index=series.index[indexes])


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
    return pd.Series(data=np.power(np.abs(np.mean(np.power(windows_strided, v), axis=1)), 1./v),
                     index=series.index[indexes])


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
    zc = np.apply_along_axis(lambda x: np.sum(np.diff(x[(x < -threshold) | (x > threshold)] > 0)), axis=1,
                             arr=windows_strided)
    return pd.Series(data=zc, index=series.index[indexes])


def feature_MNF(series, window, step):
    """Mean Frequency"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    freq, power = signal.periodogram(windows_strided, 5120)
    return pd.Series(data=np.sum(power*freq, axis=1) / np.sum(power, axis=1), index=series.index[indexes])


def feature_MDF(series, window, step):
    """Median Frequency"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    freq, power = signal.periodogram(windows_strided, 5120)
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
    freq, power = signal.periodogram(windows_strided, 5120)
    return pd.Series(data=freq[np.argmax(power, axis=1)], index=series.index[indexes])


def feature_MNP(series, window, step):
    """Mean Power"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    freq, power = signal.periodogram(windows_strided, 5120)
    return pd.Series(data=np.mean(power, axis=1), index=series.index[indexes])


def feature_TTP(series, window, step):
    """Total Power"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    freq, power = signal.periodogram(windows_strided, 5120)
    return pd.Series(data=np.sum(power, axis=1), index=series.index[indexes])


def feature_SM(series, window, step, order):
    """Spectral Moment"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    freq, power = signal.periodogram(windows_strided, 5120)
    return pd.Series(data=np.sum(power * np.power(freq, order), axis=1), index=series.index[indexes])


def feature_FR(series, window, step, flb, fhb):
    """Frequency Ratio"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    freq, power = signal.periodogram(windows_strided, 5120)
    lb = np.sum(power[:, (flb[0] < freq) & (freq < flb[1])], axis=1)
    hb = np.sum(power[:, (fhb[0] < freq) & (freq < fhb[1])], axis=1)
    return pd.Series(data=(lb / hb), index=series.index[indexes])


def feature_VCF(series, window, step):
    """Variance of Central Frequency"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    freq, power = signal.periodogram(windows_strided, 5120)

    def SM(order):
        return np.sum(power * np.power(freq, order), axis=1)

    return pd.Series(data=SM(2)/SM(0) - np.square(SM(1)/SM(0)), index=series.index[indexes])


def feature_PSR(series, window, step, n):
    """Power Spectrum Ratio"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    freq, power = signal.periodogram(windows_strided, 5120)
    PKF_id = np.argmax(power, axis=1)
    lb = np.where(PKF_id - 20 < 0, 0, PKF_id - 20)
    hb = np.where(PKF_id + 20 > window, window, PKF_id + 20)
    return pd.Series(data=[sum(p[l:h]) for p, l, h in zip(power, lb, hb)] / np.sum(power, axis=1),
                     index=series.index[indexes])


def feature_SNR(series, window, step, powerband, noiseband):
    """Signal-to-Noise Ratio"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    freq, power = signal.periodogram(windows_strided, 5120)
    snr = np.apply_along_axis(lambda p:
                              np.sum(p[(freq > powerband[0]) & (freq < powerband[1])]) /
                              (np.sum(p[(freq > noiseband[0]) & (freq < noiseband[1])]) * np.max(freq)),
                              axis=1, arr=power)
    return pd.Series(data=snr, index=series.index[indexes])


def feature_DPR(series, window, step, band, n):
    """Maximum-to-minimum Drop in Power Density Ratio"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    freq, power = signal.periodogram(windows_strided, 5120)

    dpr = pd.Series()
    for pidx in range(len(power)):
        power_b = power[pidx][(freq > band[0]) & (freq < band[1])]
        stride = power_b.strides[0]
        stride_count = len(power_b) - n + 1
        p_strided = as_strided(power_b, shape=[stride_count, n], strides=(stride, stride))
        means = np.mean(p_strided, axis=1)
        dpr.at[series.index[indexes[pidx]]] = np.max(means) / np.min(means)

    return pd.Series(data=dpr, index=series.index[indexes])


def feature_OHM(series, window, step):
    """Power Spectrum Deformation"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    freq, power = signal.periodogram(windows_strided, 5120)

    def SM(order):
        return np.sum(power * np.power(freq, order), axis=1)

    return pd.Series(data=np.sqrt(SM(2)/SM(0)) / (SM(1)/SM(0)), index=series.index[indexes])


def feature_MAX(series, window, step, order, cutoff):
    """Maximum Amplitude"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    fs = 5120
    b, a = signal.butter(order, cutoff / (0.5 * fs), btype='lowpass', analog=False, output='ba')
    return pd.Series(data=np.max(signal.lfilter(b, a, np.abs(windows_strided), axis=1), axis=1),
                     index=series.index[indexes])


def feature_SMR(series, window, step, n):
    """Signal-to-Motion Artifact Ratio"""
    # TODO: Verification Needed
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    freq, power = signal.periodogram(windows_strided, 5120)

    freq_over35 = freq > 35
    freq_over35_idx = np.argmax(freq_over35)

    smr = pd.Series()
    for pidx in range(len(power)):
        power_b = power[pidx][freq_over35]
        stride = power_b.strides[0]
        stride_count = len(power_b) - n + 1
        p_strided = as_strided(power_b, shape=[stride_count, n], strides=(stride, stride))
        mean = np.mean(p_strided, axis=1)
        max = np.max(mean)
        max_idx = np.argmax(mean) + int(np.floor(n / 2.0)) + freq_over35_idx
        a = max / freq[max_idx]

        smr.at[series.index[indexes[pidx]]] =\
            np.sum(power[pidx][freq < 600]) / np.sum(power[pidx][power[pidx] > (freq*a)])

    return pd.Series(data=smr, index=series.index[indexes])


def feature_BC(series, window, step, y_box_size_multiplier, subsampling):
    """Box-Counting Dimension"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    return pd.Series(data=np.apply_along_axis(lambda sig:
                                              ut.box_counting_dimension(sig, y_box_size_multiplier, subsampling),
                                              axis=1, arr=windows_strided), index=series.index[indexes])


def feature_PSDFD(series, window, step, power_box_size_multiplier, subsampling):
    """Power Spectral Density Fractal Dimension"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    freq, power = signal.periodogram(windows_strided, 5120)
    return pd.Series(data=np.apply_along_axis(lambda sig:
                                              ut.box_counting_dimension(sig, power_box_size_multiplier, subsampling),
                                              axis=1, arr=power), index=series.index[indexes])