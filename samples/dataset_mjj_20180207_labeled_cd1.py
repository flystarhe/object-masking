import sys
sys.path.insert(0, "/data2/gits/object-masking")


import os
import json
import collections
from imgaug import augmenters as iaa
# import object_masking
from object_masking import utils_annotations
from object_masking import utils_mask_rcnn
from object_masking import model_mask_rcnn


def dataset_labelme_to_via(source, output):
    jsonfile = utils_annotations.labelme_to_via(source, output)
    print(jsonfile)
    result = utils_annotations.dataset_split(jsonfile, output_dir=output)
    print("\n".join(result))


def dataset_classes(dataset_dir, subset, jsonfile="via_region_data.json"):
    classes = collections.defaultdict(int)

    with open(os.path.join(dataset_dir, subset, jsonfile)) as f:
        annotations = json.load(f).values()

    for a in annotations:
        regions = a.get("regions", None)
        if regions is None:
            print("warning: not find regions [{}]".format(a["filename"]))
            continue
        for region in regions.values():
            classes[region["region_attributes"].get("name", "**")] += 1

    return classes


if __name__ == "__main__":
    source = "/data2/datasets/slyx/mjj_20180207/labeled_cd1/dataset00"
    output = "/data2/datasets/slyx/mjj_20180207/labeled_cd1/dataset00"
    dataset_labelme_to_via(source, output)
    #'/data2/datasets/slyx/mjj_20180207/labeled_cd1/dataset00/via_region_data.json'
    #('/data2/datasets/slyx/mjj_20180207/labeled_cd1/dataset00/via_region_data_train.json',
    # '/data2/datasets/slyx/mjj_20180207/labeled_cd1/dataset00/via_region_data_val.json')

    ## prepare dataset
    root = "/data2/datasets/slyx/mjj_20180207/labeled_cd1/dataset00"
    classes = [("vgg_via", 1, "ggo")]
    dataset = model_mask_rcnn.get_dataset(root, "", classes, jsonfile="via_region_data_val.json")
    ## view dataset_masks
    model_mask_rcnn.dataset_display_masks(dataset)
    ## view dataset_boxes
    model_mask_rcnn.dataset_display_boxes(dataset)

    ## image augmentation
    mydict = {"NUM_CLASSES": 2,
              "IMAGE_MIN_DIM": 512,
              "IMAGE_MAX_DIM": 512}
    config = model_mask_rcnn.MyConfig(mydict, "none", 1, 4)
    augmentation = [iaa.Fliplr(0.5), iaa.Affine(scale=(0.8, 1.2), order=0), iaa.Affine(rotate=(-10, 10), order=0)]
    augmentation = iaa.SomeOf((0, None), augmentation)
    augmentation = iaa.Sometimes(0.5, augmentation)
    model_mask_rcnn.dataset_display_instances(config, dataset, augmentation=augmentation)