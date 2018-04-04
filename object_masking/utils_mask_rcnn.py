import sys
MASK_RCNN = "/data2/gits/object-masking/modules/mask_rcnn"
if MASK_RCNN not in sys.path:
    sys.path.insert(0, MASK_RCNN)


import os
import json
import numpy as np
import skimage
import skimage.draw
# import mask_rcnn
import utils


################################################################
#  Dataset
################################################################
class MyDataset(utils.Dataset):
    def load_dataset(self, dataset_dir, subset, classes=None, format="vgg_via"):
        if format == "vgg_via":
            return self.load_dataset_vgg_via(dataset_dir, subset, classes)
        else:
            raise ValueError("Unsupported dataset format: {}".format(format))

    def load_dataset_vgg_via(self, dataset_dir, subset, classes=None, jsonfile=None):
        """Load a subset of the VGG_VIA dataset.
        dataset_dir: Root directory of the dataset
        subset: String, directory name of the subset
        classes: `[(source:str, class_id:int, class_name:str),..]`
        """
        if classes is None:
            classes = [("vgg_via", 1, "none")]

        if jsonfile is None:
            jsonfile = "via_region_data.json"

        # Add classes
        for item in classes:
            self.add_class(*item)

        class_names = set(c[2] for c in classes)

        assert subset in ["train", "val", ""]
        dataset_dir = os.path.join(dataset_dir, subset)

        with open(os.path.join(dataset_dir, jsonfile)) as f:
            annotations = json.load(f).values()
            annotations = [a for a in annotations if a["regions"]]

        def good_region(region, class_names):
            region_attributes = region["region_attributes"]
            if region_attributes.get("name", "none").strip().lower() in class_names:
                return True
            return False

        # Add images
        for a in annotations:
            image_path = os.path.join(dataset_dir, a["filename"])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            regions = [r for r in a["regions"].values() if good_region(r, class_names)]
            kwargs = {"height": height, "width": width, "regions": regions}
            self.add_image("vgg_via", image_id=a["filename"], path=image_path, **kwargs)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
          masks: A bool array of shape [height, width, instance count]
          class_ids: a 1D array of class IDs of the instance masks
        """
        info = self.image_info[image_id]

        if info["source"] == "vgg_via":
            masks = np.zeros([info["height"], info["width"], len(info["regions"])], dtype=np.uint8)
            class_ids = np.zeros([len(info["regions"])], dtype=np.int32)

            for i, region in enumerate(info["regions"]):
                shape_attributes = region["shape_attributes"]
                region_attributes = region["region_attributes"]

                if shape_attributes["name"] == "rect":
                    x, y = shape_attributes["x"], shape_attributes["y"]
                    w, h = shape_attributes["width"], shape_attributes["height"]
                    masks[y:y+h, x:x+w, i] = 1
                elif shape_attributes["name"] == "polygon":
                    rr, cc = skimage.draw.polygon(shape_attributes["all_points_y"], shape_attributes["all_points_x"])
                    masks[rr, cc, i] = 1
                elif shape_attributes["name"] == "circle":
                    rr, cc = skimage.draw.circle(shape_attributes["cy"], shape_attributes["cx"], shape_attributes["r"])
                    masks[rr, cc, i] = 1
                elif shape_attributes["name"] == "ellipse":
                    rr, cc = skimage.draw.ellipse(shape_attributes["cy"], shape_attributes["cx"], shape_attributes["ry"], shape_attributes["rx"])
                    masks[rr, cc, i] = 1

                class_ids[i] = self.class_names.index(region_attributes.get("name", "none").strip().lower())

            return masks, class_ids
        else:
            return super(self.__class__, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]

        if info["source"] == "vgg_via":
            return info["path"]
        else:
            return super(self.__class__, self).image_reference(image_id)