# putemg-feature-extractor
Dedicated EMG feature calculator for putEMG Dataset http://biolab.put.poznan.pl/putemg-dataset/

## Usage
### Script
putemg-feature-extractor script will calculate features of given putEMG record in HDF5 format based on feature decriptors in given XML file. See "all_features.xml" for file format and feature list along with its parameters. Output will be written to corresponding HDF5 format file with "_features" sufix.

```bash
putemg-feature-extractor.py <feature_config_file> <putemg_hdf5_file>
```

Arguments:

`<feature_config_file>` - URL to XML file containing feature descriptors

`<putemg_hdf5_file>` - URL of putEMG HDF5 files containing experiment data

### Module
putemg-feature-extractor can be also used as Python3 module. In order to calculate features of a given hdf5 putEMG file based on XML feature descriptor file see example below. See "all_features.xml" for file format and feature list along with its parameters.

```python
import features

xml_url = 'all_features.xml'
hdf5_url = 'emg_gestures-03-repeats_long-2018-05-11-11-05-00-695.hdf5'
ft = features.features_from_xml(xml_url, hdf5_url)
```

It is also possible to calculate desired single feature directly on already imported putEMG record. Avaiable features and its parameters are same as in "all_features.xml" file. Eg.:
```python
import features as ft
import pandas as pd

record = pd.read_hdf('emg_gestures-03-repeats_long-2018-05-11-11-05-00-695.hdf5')
df1 = ft.calculate_feature(record, 'ZC', window=1000, step=500, threshold=30)
df2 = ft.calculate_feature(record[22.5:30.9], name='RMS', window=500, step=250)
df3 = ft.calculate_feature(record[1:10][['EMG_1', 'EMG_5']], 'RMS', window=500, step=250)
```

## Feature List
* Integral Absolute Value (IAV)
* Average Amplitude Change (AAC)
* Approximate Entropy (ApEn)
* Auto-Regressive Coefficients (AR)
* Cepstral Coefficients (CC)
* Difference Absolute Standard Deviation Value (DASDV)
* Kurtosis (Kurt)
* LOG Detector (LOG)
* Modified Mean Absolute Value Type 1 (MAV1)
* Modified Mean Absolute Value Type 2 (MAV2)
* Mean Absolute Value (MAV)
* Mean Absolute Value Slope (MAVSLP)
* Multiple Hamming Windows (MHW)
* Multiple Trapezoidal Windows (MTW)
* Myopulse Percentage Rate (MYOP)
* Root Mean Square (RMS)
* Sample Entropy (SampEn)
* Skewness (Skew)
* Slope Sign Change (SSC)
* Simple Square Integral (SSI)
* Absolute Temporal Moment (TM)
* Variance (VAR)
* V-Order (V)
* Willison Amplitude (WAMP)
* Waveform Length (WL)
* Zero Crossing (ZC)
* Mean Frequency (MNF)
* Median Frequency (MDF)
* Peak Frequency (PKF)
* Mean Power (MNP)
* Total Power (TTP)
* Frequency Ratio (FR)
* Variance of Central Frequency (VCF)
* Power Spectrum Ratio (PSR)
* Signal-to-Noise Ratio (SNR)
* Maximum-to-minimum Drop in Power Density Ratio (DPR)
* Power Spectrum Deformation (OHM)
* Maximum Amplitude (MAX)
* Signal-to-Motion Artifact Ratio (SMR)
* Box-Counting Dimension (BC)

## Dependencies
* Pandas - https://pandas.pydata.org/
* Numpy - http://www.numpy.org/
* SciPy - https://www.scipy.org/

## Attributions
* PyEEG v0.4.0 - SampEn and ApEn features - GNU GLP v3 - http://pyeeg.org