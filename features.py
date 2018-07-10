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


def feature_ZeroCross(series, window, step, deadzone):
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    zc = np.apply_along_axis(lambda x: np.sum(np.diff(x[(x < -deadzone) | (x > deadzone)] > 0)), axis=1, arr=windows_strided)
    return pd.Series(data=zc, index=series.index[indexes])


def feature_RMS(series, window, step):
    windows_strided, indexes = ut.moving_window_stride(series.values, window, step)
    return pd.Series(data=np.sqrt(np.mean(np.square(windows_strided), axis=1)), index=series.index[indexes])
