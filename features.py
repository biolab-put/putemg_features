import pandas as pd
import numpy as np
import utilities as ut
import xml.etree.ElementTree as ET


def calculate_feature(record, params):
    """
    Calculates feature given by params['name'] of given pandas.DataFrame. Feature is calculated for each Series with
    column name of "EMG_\d". Feature parameters are passed by Dict params, eg. {'name': RMS, 'window': 500, 'step': 250}
    :param record: pandas.DataFrame - input DataFrame with data to calculate features from
    :param params: Dict - parameter dictionary for feature calculation. Must contain at least 'name' entry
    :return: pandas.DataFrame  - DataFrame containing output of desired feature
    """
    feature_func_name = 'feature_' + params['name']  # Get feature function name based on name
    feature_values = pd.DataFrame()  # Create empty DataFrame

    for column in record.filter(regex="EMG_\d").columns:  # For each column containing EMG data (for each Series)
        feature_label = params['name'] + '_' + column.split('_')[1]  # Prepare feature column label
        feature_values[feature_label] = globals()[feature_func_name](record[column], params)  # Call feature calculation by function name, and add to output DataFtame

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
        feature_frame = feature_frame.join(calculate_feature(record, xml_entry.attrib), how="outer")  # add to output frame values calculated by each feature function

    return feature_frame


def feature_ZeroCross(series, params):
    print("ZeroCross")
    print(params)
    return pd.Series(np.random.randn(20))


def feature_RMS(series, params):
    print("RMS")
    print(params)
    return pd.Series(np.random.randn(10))


if __name__ == '__main__':
    hdf5 = 'P:\\Data-HDF5\\emg_gestures-01-repeats_short-2018-05-08-15-06-32-389.hdf5'
    xml = 'all_features.xml'

    # r_hdf5 = pd.read_hdf(hdf5)
    # print(calculate_feature(r_hdf5, {'name': 'ZeroCross', 'window': 500, 'step': 250}))
    print(features_from_xml(xml, hdf5))
