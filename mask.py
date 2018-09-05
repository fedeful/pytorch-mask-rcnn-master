import os
import torch
import pickle
import cv2
import coco
import skimage.io
import model as modellib
import numpy as np


def masks_generation(train, source, destination, all=True):
    print('START PROGRAM !!!')

    ROOT_DIR = os.getcwd()
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.pth")
    IMAGE_DIR = os.path.join(ROOT_DIR, "images")

    class InferenceConfig(coco.CocoConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        # GPU_COUNT = 0 for CPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    config.display()

    # Create model object.
    model = modellib.MaskRCNN(model_dir=MODEL_DIR, config=config)
    if config.GPU_COUNT:
        model = model.cuda()

    # Load weights trained on MS-COCO
    model.load_state_dict(torch.load(COCO_MODEL_PATH))

    # COCO Class names
    # Index of the class in the list is its ID. For example, to get ID of
    # the teddy bear class, use: class_names.index('teddy bear')
    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                   'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                   'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush']

    counter = 0
    invalid_file = []

    if not os.path.isdir(destination):
        os.mkdir(destination, 0o777)

    for fname in os.listdir(source):
        counter = counter + 1

        if not train and (fname.split('_')[0] == '-1' or fname.split('_')[0] == '0000'):
            continue

        #image = skimage.io.imread(os.path.join(source, fname))
        image = cv2.imread(os.path.join(source, fname), cv2.IMREAD_COLOR)
        if image is None:
            invalid_file.append(fname)
            continue
        if len(image.shape) != 3:
            invalid_file.append(fname)
            continue
        if image.shape[2] != 3:
            invalid_file.append(fname)
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = model.detect([image])
        if results is None:
            invalid_file.append(fname)
            print('None File')
            continue
        r = results[0]

        person = []

        for k, v in enumerate(r['class_ids']):
            if v == 1:
                person.append(k)

        if len(person) != 0:
            if all:
                a = np.zeros([128, 64], np.uint8)
            else:
                M_SCORE = 0
                K_MAX = 0
                for k, v in enumerate(r['scores']):
                    if k in person and v > M_SCORE:
                        M_SCORE = v
                        K_MAX = k

                final_image = (r['masks'][:, :, K_MAX] * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(destination, fname), final_image)
        else:
            invalid_file.append(fname)
            continue

        if counter != 0 and counter % 500 == 0:
            print(counter)

    if train:
        with open('./no_mask_train.pck', 'wb') as fn:
            pickle.dump(invalid_file, fn)
    else:
        with open('./no_mask_test.pck', 'wb') as fn:
            pickle.dump(invalid_file, fn)


if __name__ == '__main__':
    SOURCE_TRAIN = '/homes/ffulgeri/Datasets/Market-1501-v15.09.15/bounding_box_train'
    SOURCE_TEST = '/homes/ffulgeri/Datasets/Market-1501-v15.09.15/bounding_box_train'
    DESTINATION = '/tmp/fedemarket'
    masks_generation(True, SOURCE_TRAIN, DESTINATION, all=False)
