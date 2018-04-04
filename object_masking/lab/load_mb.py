import os
from pyhej.image.dicom import read_dicominfo


def get_pairs(data_path, name=None):
    data = []

    for i, ipath in enumerate(sorted(os.listdir(data_path)), 1):
        if not os.path.isdir(os.path.join(data_path, ipath)):
            continue

        for j, jpath in enumerate(sorted(os.listdir(os.path.join(data_path, ipath))), 1):
            filename = os.path.join(data_path, ipath, jpath)
            vals, plan = read_dicominfo(filename)
            if vals:
                vals['Group'] = str(name) + '/' + ipath
                data.append(vals)

    return data