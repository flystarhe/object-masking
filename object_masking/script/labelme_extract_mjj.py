import sys
Libs = ["/data2/gits/object-masking/modules/pyhej"]
for item in Libs:
    if item not in sys.path:
        sys.path.insert(0, item)


import shutil
from pathlib import Path
# import pyhej
from pyhej.utils import set_dir


def extract_dataset(root=None, anno_dir=None, data_dir=None, save_dir="dataset00"):
    if root is not None:
        root = Path(root)
        anno_dir = root/"Annotations"
        data_dir = root/"Images"
        save_dir = root/save_dir
    else:
        anno_dir = Path(anno_dir)
        data_dir = Path(data_dir)
        save_dir = Path(save_dir)

    set_dir(str(save_dir), rm=True)

    for item in anno_dir.glob("**/*.xml"):
        tmppath = item.relative_to(anno_dir)
        xmlpath = (anno_dir/tmppath).with_suffix(".xml")
        jpgpath = (data_dir/tmppath).with_suffix(".jpg")
        tmppath = "_".join(tmppath.parts)
        dst_xml = (save_dir/tmppath).with_suffix(".xml")
        dst_jpg = (save_dir/tmppath).with_suffix(".jpg")
        shutil.copy(xmlpath.as_posix(), dst_xml.as_posix())
        shutil.copy(jpgpath.as_posix(), dst_jpg.as_posix())

    return str(save_dir)


if __name__ == "__main__":
    root = "/data2/datasets/slyx/mjj_20180207/labeled_cd1"
    res = extract_dataset(root, save_dir=sys.argv[1])
    print(res)