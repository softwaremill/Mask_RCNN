import os
import sys
import xml.etree.ElementTree as ET
from plyfile import PlyData
import numpy as np

file_dirname = os.path.dirname(__file__)
ROOT_DIR = os.path.join(file_dirname, "../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn.utils import Dataset
from mrcnn import model as modellib, utils

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


class SceneNNConfig(Config):
    """Configuration for training on SceneNN.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "scenenn"

    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 40


class SceneNNDataset(Dataset):

    def load_scenenn(self, dataset_dir, subset):
        """Load a subset of the SceneNN dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes.
        tree = ET.parse(os.path.join(dataset_dir, 'nyu_color.xml'))
        root = tree.getroot()

        classes = {}
        for child in root:
            classes[child.attrib['text']] = child.attrib['id']
            self.add_class("scenenn", child.attrib['id'], child.attrib['text'])

        # Train or validation dataset?
        assert subset in ["train", "val", "train_small", "val_small"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        annotations = {}
        for image_dir in os.listdir(dataset_dir):
            dir_tree = ET.parse(os.path.join(dataset_dir, image_dir, f"{image_dir}.xml"))
            root = dir_tree.getroot()
            objects = {}
            for child in root:
                objects[child.attrib['id']] = child.attrib
            annotations[image_dir] = objects

        # Add images
        for key, value in annotations.items():
            polygons = {}

            image_path = os.path.join(dataset_dir, key, f"{key}.ply")
            plydata = PlyData.read(image_path)

            vertex = plydata['vertex']

            xs = (vertex['x'] * 100).astype(np.int)
            ys = (vertex['y'] * 100).astype(np.int)
            zs = (vertex['z'] * 100).astype(np.int)

            min_x = xs.min()
            min_y = ys.min()
            min_z = zs.min()

            min_overall = min([min_x, min_y, min_z])

            if min_overall < 0:
                xs = xs - min_overall
                ys = ys - min_overall
                zs = zs - min_overall

            labels = vertex['label']

            height, width, depth = xs.max(), ys.max(), zs.max()

            # class_id = classes[value['nyu_class']]

            for label in labels:
                if not value[str(label)]['nyu_class'] == 'unknown':
                    class_id = classes[value[str(label)]['nyu_class']]
                    polygons[label] = {'points': [], 'class_id': class_id}

            for x, y, z, label in zip(xs, ys, zs, labels):
                if not value[str(label)]['nyu_class'] == 'unknown':
                    polygons[label]['points'].append((x, y, z))

            self.add_image(
                "scenenn",
                image_id=key,  # use file name as a unique image id
                path=image_path,
                width=width, height=height, depth=depth,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a scenenn dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "scenenn":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"] + 1, info["width"] + 1, info["depth"] + 1, len(info["polygons"])],
                        dtype=np.uint8)

        class_ids = []
        for i, p in enumerate(info["polygons"].values()):
            # Get indexes of pixels inside the polygon and set them to 1
            class_ids.append(p['class_id'])
            xs, ys, zs = zip(*p['points'])
            mask[xs, ys, zs, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "balloon":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = SceneNNDataset()
    dataset_train.load_scenenn(args.dataset, "train_small")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = SceneNNDataset()
    dataset_val.load_scenenn(args.dataset, "val_small")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect scenenn objects.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=False,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = SceneNNConfig()
    else:
        class InferenceConfig(SceneNNConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    # if args.weights.lower() == "coco":
    #     weights_path = COCO_WEIGHTS_PATH
    #     # Download weights file
    #     if not os.path.exists(weights_path):
    #         utils.download_trained_weights(weights_path)
    # elif args.weights.lower() == "last":
    #     # Find last trained weights
    #     weights_path = model.find_last()
    # elif args.weights.lower() == "imagenet":
    #     # Start from ImageNet trained weights
    #     weights_path = model.get_imagenet_weights()
    # else:
    #     weights_path = args.weights
    #
    # # Load weights
    # print("Loading weights ", weights_path)
    # if args.weights.lower() == "coco":
    #     # Exclude the last layers because they require a matching
    #     # number of classes
    #     model.load_weights(weights_path, by_name=True, exclude=[
    #         "mrcnn_class_logits", "mrcnn_bbox_fc",
    #         "mrcnn_bbox", "mrcnn_mask"])
    # else:
    #     model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)

    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))