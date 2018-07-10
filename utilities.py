import ast


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
