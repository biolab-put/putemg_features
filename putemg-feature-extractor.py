import os
import sys


def usage():
    print()
    print('Usage: {:s} <feature_config_file> <putemg_hdf5_folder>'.format(os.path.basename(__file__)))
    print()
    print('Arguments:')
    print('    <feature_config_file>   URL to XML file containing feature descriptors')
    print('    <putemg_hdf5_folder>    URL of putEMG Dataset folder containing HDF5 files')
    print()
    print('Examples:')
    print('{:s} all_features.xml ./putEMG/Data-HDF5'.format(os.path.basename(__file__)))
    exit(1)


if len(sys.argv) < 3:
    print('Illegal number of parameters')
    usage()

xml_file_url = sys.argv[1]
if not os.path.isfile(xml_file_url):
    print('XML file with feature descriptos does not exist - {:s}'.format(xml_file_url))
    usage()

hdf5_folder_url = sys.argv[2]
if not os.path.isdir(hdf5_folder_url):
    print('putEMG HDF5 folder does not exist - {:s}'.format(hdf5_folder_url))
    usage()

# 1) get hdf5 folder as argument
# 2) list the files to extract features from
# 3) read xml feature config file
# 4)
