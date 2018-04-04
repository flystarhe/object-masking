import sys
Libs = ["/data2/gits/object-masking/modules/pyhej"]
for item in Libs:
    if item not in sys.path:
        sys.path.insert(0, item)


import os
import json
import codecs
from pathlib import Path
# import pyhej
from pyhej.utils import set_dir, set_parent
from pyhej.utils.xml import parser_file, parser_labelme


def labelme_to_via(source_dir, output_dir=None):
    dataset = {}
    for item in sorted(Path(source_dir).glob("*.xml")):
        filename = item.with_suffix(".jpg").name
        annotation = parser_file(item.as_posix())["annotation"]
        objects = annotation["object"]

        if not isinstance(objects, list):
            objects = [objects]

        regions = {}
        for obj in objects:
            if "name" in obj and "polygon" in obj:
                region_attributes = {"name": obj.get("name", "none")}
                if obj.get("type") == "bounding_box":
                    pt1 = obj["polygon"]["pt"][0]
                    pt3 = obj["polygon"]["pt"][2]
                    shape_attributes = {"x": int(pt1["x"]),
                                        "y": int(pt1["y"]),
                                        "width": int(pt3["x"]) - int(pt1["x"]),
                                        "height": int(pt3["y"]) - int(pt1["y"]),
                                        "name": "rect"}
                else:
                    xs, ys = [], []
                    for pt in obj["polygon"]["pt"]:
                        xs.append(int(pt["x"]))
                        ys.append(int(pt["y"]))
                    shape_attributes = {"all_points_x": xs,
                                        "all_points_y": ys,
                                        "name": "polygon"}
                regions[len(regions)] = {"region_attributes": region_attributes, "shape_attributes": shape_attributes}

        dataset[filename] = {"filename": filename, "regions": regions}

    if output_dir:
        set_dir(output_dir)
        filepath = os.path.join(output_dir, "via_region_data.json")
        with codecs.open(filepath, "w", "utf-8") as writer:
            writer.write(json.dumps(dataset))
        return filepath
    else:
        return dataset


def merge_via_files(source_dir, output_dir=None, pattern="via_region_data_sub*.json"):
    dataset = {}
    for item in sorted(Path(source_dir).glob(pattern)):
        item = item.as_posix()
        print("merging file {} ..".format(item))
        with codecs.open(item, "r", "utf-8") as reader:
            annotations = json.load(reader).values()
        for annotation in annotations:
            filename = annotation["filename"]
            if filename in dataset:
                print("Warning!!! replace {} value from {}!".format(filename, item))
            dataset[filename] = annotation

    assert dataset, "not find {}".format(pattern)

    if output_dir:
        set_dir(output_dir)
        filepath = os.path.join(output_dir, "via_region_data.json")
        with codecs.open(filepath, "w", "utf-8") as writer:
            writer.write(json.dumps(dataset))
        return filepath
    else:
        return dataset


def dataset_split(jsonfile, val_size=0.2, shuffle=False, output_dir=None):
    dataset_val = {}
    dataset_train = {}

    with codecs.open(jsonfile, "r", "utf-8") as reader:
        annotations = json.load(reader).values()
        annotations = list(annotations)

    if val_size < 1.0:
        val_size = int(val_size*len(annotations))

    if shuffle:
        random.shuffle(annotations)

    for annotation in annotations[:val_size]:
        dataset_val[annotation["filename"]] = annotation

    for annotation in annotations[val_size:]:
        dataset_train[annotation["filename"]] = annotation

    if output_dir:
        set_dir(output_dir)
        filepath_val = os.path.join(output_dir, "via_region_data_val.json")
        with codecs.open(filepath_val, "w", "utf-8") as writer:
            writer.write(json.dumps(dataset_val))
        filepath_train = os.path.join(output_dir, "via_region_data_train.json")
        with codecs.open(filepath_train, "w", "utf-8") as writer:
            writer.write(json.dumps(dataset_train))
        return filepath_train, filepath_val
    else:
        return dataset_train, dataset_val