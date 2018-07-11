import pandas as pd
import numpy as np
import utilities as ut
import xml.etree.ElementTree as ET


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


def feature_DASDV(series, window, step):
    """Difference Absolute Standard Deviation Value"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    return pd.Series(data=np.sqrt(np.mean(np.square(np.diff(windows_strided)), axis=1)), index=series.index[indexes])


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


def feature_WL(series, window, step):
    """Waveform Length"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    return pd.Series(data=np.sum(np.diff(windows_strided), axis=1), index=series.index[indexes])


def feature_ZC(series, window, step, deadzone):
    """Zero Crossing"""
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    zc = np.apply_along_axis(lambda x: np.sum(np.diff(x[(x < -deadzone) | (x > deadzone)] > 0)), axis=1, arr=windows_strided)
    return pd.Series(data=zc, index=series.index[indexes])
