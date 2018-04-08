import sys
Libs = ["/data2/gits/object-masking",
        "/data2/gits/object-masking/modules/mask_rcnn"]
for item in Libs:
    if item not in sys.path:
        sys.path.insert(0, item)


import pprint
import numpy as np
import skimage
from keras.utils.data_utils import get_file
# import mask_rcnn
import mrcnn.utils as mrcnn_utils
import mrcnn.visualize as mrcnn_visualize
import mrcnn.config as mrcnn_config
import mrcnn.model as mrcnn_model
# import object_masking
from object_masking.utils_mask_rcnn import MyDataset


class MyConfig(mrcnn_config.Config):
    def __init__(self, mydict, name="hej", gpu_count=1, images_per_gpu=1):
        # BATCH_SIZE = IMAGES_PER_GPU * GPU_COUNT
        # NUM_CLASSES = (1)Background + (?)NUM_CLASSES
        self.NAME = name
        self.GPU_COUNT = gpu_count
        self.IMAGES_PER_GPU = images_per_gpu
        self.NUM_CLASSES = mydict.get("NUM_CLASSES", 2)
        self.IMAGE_MIN_DIM = mydict.get("IMAGE_MIN_DIM", 800)
        self.IMAGE_MAX_DIM = mydict.get("IMAGE_MAX_DIM", 1024)
        super(MyConfig, self).__init__()


def get_model(mode, config, model_dir, init_with="coco"):
    """
    mode: Either "training" or "inference"
    config: A Sub-class of the Config class
    model_dir: Directory to save training logs and trained weights
    init_with: imagenet, coco, last, or model_path
    """
    model = mrcnn_model.MaskRCNN(mode, config, model_dir)
    if init_with == "imagenet":
        model_path = model.get_imagenet_weights()
        model.load_weights(model_path, by_name=True)
    elif init_with == "coco":
        model_path = "https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5"
        model_path = get_file("mask_rcnn_coco.h5", model_path, cache_subdir="models")
        model_exclude = ["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"]
        model.load_weights(model_path, by_name=True, exclude=model_exclude)
    elif init_with == "last":
        model_path = model.find_last()[1]
        model.load_weights(model_path, by_name=True)
    else:
        model_path = init_with
        model.load_weights(model_path, by_name=True)
    return model


def get_dataset(dataset_dir, subset, classes, jsonfile=None):
    """
    dataset_dir: Root directory of the dataset
    subset: String, directory name of the subset
    classes: `[(source, class_id, class_name),..]`
    """
    data = MyDataset()
    data.load_dataset_vgg_via(dataset_dir, subset, classes, jsonfile)
    data.prepare()
    return data


def dataset_display_masks(dataset, image_ids=None, limit=4):
    if image_ids is None:
        #np.random.choice(dataset.image_ids, 10)
        image_ids = dataset.image_ids
    for image_id in image_ids:
        image = dataset.load_image(image_id)
        mask, class_ids = dataset.load_mask(image_id)
        print(dataset.image_reference(image_id))
        mrcnn_visualize.display_top_masks(image, mask, class_ids, dataset.class_names, limit)


def dataset_display_boxes(dataset, image_ids=None):
    if image_ids is None:
        #np.random.choice(dataset.image_ids, 10)
        image_ids = dataset.image_ids
    for image_id in image_ids:
        image = dataset.load_image(image_id)
        mask, class_ids = dataset.load_mask(image_id)
        # Compute Bounding box
        bbox = mrcnn_utils.extract_bboxes(mask)
        print(dataset.image_reference(image_id))
        mrcnn_visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)


def dataset_display_instances(config, dataset, image_ids=None, augmentation=None):
    if image_ids is None:
        #np.random.choice(dataset.image_ids, 10)
        image_ids = dataset.image_ids
    for image_id in image_ids:
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            mrcnn_model.load_image_gt(dataset, config, image_id, augmentation=augmentation, use_mini_mask=False)
        print(dataset.image_reference(image_id))
        mrcnn_model.log("image", image)
        mrcnn_model.log("image_meta", image_meta)
        mrcnn_model.log("gt_class_id", gt_class_id)
        mrcnn_model.log("gt_bbox", gt_bbox)
        mrcnn_model.log("gt_mask", gt_mask)
        mrcnn_visualize.display_instances(image, gt_bbox, gt_mask, gt_class_id, dataset.class_names)


def training(model, dataset_train, dataset_val, stages=None):
    """
    dataset_train, dataset_val: Training and validation Dataset objects.
    stages: Training network by stage `[{"learning_rate":, "epochs":, "layers":, "augmentation":}]`
    layers: Allows selecting wich layers to train. It can be:
        - A regular expression to match layer names to train
        - One of these predefined values:
            heads: The RPN, classifier and mask heads of the network
            all: All the layers
            3+: Train Resnet stage 3 and up
            4+: Train Resnet stage 4 and up
            5+: Train Resnet stage 5 and up
    """
    if stages is None:
        stages = [{"learning_rate": 0.001, "epochs": 50, "layers": "heads"}]

    for stage in stages:
        print("=> Training network:\n{}".format(stage))
        model.train(dataset_train, dataset_val, **stage)


def detect(model, config, images, verbose=0):
    """
    images: List of images, potentially of different sizes
        - `model.detect(images, verbose)` require `len(images) == BATCH_SIZE`
    """
    results = []
    for image in images:
        if isinstance(image, str):
            image = skimage.io.imread(image)
            if image.ndim != 3:
                image = skimage.color.gray2rgb(image)
            if image.shape[-1] == 4:
                image = image[..., :3]
        results.append(model.detect([image], verbose)[0])
    return results


def detect_display_masks(image, r, class_names, limit=4):
    if isinstance(image, str):
        image = skimage.io.imread(image)
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        if image.shape[-1] == 4:
            image = image[..., :3]
    mrcnn_visualize.display_top_masks(image, r["masks"], r["class_ids"], class_names, limit)


def detect_display_instances(image, r, class_names):
    if isinstance(image, str):
        image = skimage.io.imread(image)
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        if image.shape[-1] == 4:
            image = image[..., :3]
    mrcnn_visualize.display_instances(image, r["rois"], r["masks"], r["class_ids"], class_names, scores=r["scores"])


def evaluation(model, config, dataset, image_ids=None, verbose=0):
    APs = []
    if image_ids is None:
        #np.random.choice(dataset.image_ids, 10)
        image_ids = dataset.image_ids
    for image_id in image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            mrcnn_model.load_image_gt(dataset, config, image_id, use_mini_mask=False)
        molded_images = np.expand_dims(mrcnn_model.mold_image(image, config), 0)
        # Run object detection
        results = model.detect([image], verbose)
        r = results[0]
        # Compute AP
        if r["class_ids"].size == 0:
            continue
        AP, precisions, recalls, overlaps =\
            mrcnn_utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r["masks"])
        APs.append(AP)
    return APs, np.mean(APs)


def demo_script_training():
    classes = [("vgg_via", 1, "ggo")]
    dataset_dir = "/data2/datasets/slyx/mjj_20180207/labeled_cd1/dataset00"
    dataset_val = get_dataset(dataset_dir, "", classes, "via_region_data_val.json")
    dataset_train = get_dataset(dataset_dir, "", classes, "via_region_data_train.json")

    # pip install imgaug
    from imgaug import augmenters as iaa
    augmentation = [iaa.Fliplr(0.5), iaa.Affine(scale=(0.8, 1.2), order=0), iaa.Affine(rotate=(-10, 10), order=0)]
    augmentation = iaa.SomeOf((0, None), augmentation)
    augmentation = iaa.Sometimes(0.5, augmentation)

    mydict = {"NUM_CLASSES": 2,
              "IMAGE_MIN_DIM": 512,
              "IMAGE_MAX_DIM": 512}
    config = MyConfig(mydict, "none", 1, 4)
    model = get_model("training", config, "tmps", init_with="coco")

    stages = [{"learning_rate": 0.001, "epochs": 50, "layers": "heads", "augmentation": augmentation},
              {"learning_rate": 0.001, "epochs": 100, "layers": "4+", "augmentation": augmentation},
              {"learning_rate": 0.0001, "epochs": 150, "layers": "all", "augmentation": augmentation}]

    training(model, dataset_train, dataset_val, stages)


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train Mask R-CNN ..")
    parser.add_argument("command", metavar="<command>",
                        help="`train` or `test`")
    parser.add_argument("--dataset", required=False, metavar="/path/to/dataset/",
                        help="Directory of the dataset")
    parser.add_argument("--weights", required=True, metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or `coco` or `last` or `imagenet`")
    parser.add_argument("--logs", required=False, default="tmps/logs", metavar="/path/to/logs/",
                        help="Logs and checkpoints directory (default=logs/)")
    parser.add_argument("--image", required=False, metavar="path or URL to image",
                        help="Image to apply the model on")
    parser.add_argument("--video", required=False, metavar="path or URL to video",
                        help="Video to apply the model on")
    #args = parser.parse_args("train --dataset=tmps/dataset --weights=last".split())
    args = parser.parse_args()
    pprint.pprint(args.__dict__)

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "test":
        assert args.image or args.video, "Provide --image or --video to apply"