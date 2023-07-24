"""
Format of label

images: id, width, height, file_name, seg_file_name

categories: id, name
{'id': 1, 'name': 'car'},
{'id': 2, 'name': 'truck'},
{'id': 3, 'name': 'bicycle'},
{'id': 4, 'name': 'person'},
{'id': 5, 'name': 'motorcycle'},
{'id': 6, 'name': 'bus'},
{'id': 7, 'name': 'rider'},
{'id': 8, 'name': 'train'}

annatations: id, image_id, segmentation, category_id, iscrowd, area, bbox

- data
 - cityscapes
    - train
      - image
      - depth
      - train.json
    - val
      -image
      - depth
      - val.json
"""


from torch.utils.data.dataset import Dataset
import json
import os
import numpy as np
from PIL import Image
import cv2
import torch
Root = "/content/drive/MyDrive/cityscapes-to-coco-conversion-master"


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


def preprocess_input(inputs):
    MEANS = (104, 117, 123)
    return inputs - MEANS


class CityScapes(Dataset):
    def __init__(self, train_lines, input_shape, anchors, batch_size, num_classes, is_train):
        super(CityScapes, self).__init__()

        # self.length = len(self.data['images'])
        self.input_shape = input_shape
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.is_train = is_train
        self.overlap_threshold = 0.8
        self.train_lines = train_lines
        self.length = len(self.train_lines)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length

        image, box, disparity = self.get_random_data(
            index, self.input_shape, random=self.is_train)
        image_data = np.transpose(preprocess_input(
            np.array(image, dtype=np.float32)), (2, 0, 1))
        disparity[disparity == 0] = -1

        if len(box) != 0:
            boxes = np.array(box[:, :4], dtype=np.float32)

            boxes[:, [0, 2]] = boxes[:, [0, 2]] / self.input_shape[1]
            boxes[:, [1, 3]] = boxes[:, [1, 3]] / self.input_shape[0]

            one_hot_label = np.eye(
                self.num_classes - 1)[np.array(box[:, 4], np.int32)]
            box = np.concatenate([boxes, one_hot_label], axis=-1)

        box = self.assign_boxes(box)
        image_data = torch.from_numpy(image_data).float()
        disparity = torch.from_numpy(np.array(disparity)).unsqueeze(0).float()
        box = torch.from_numpy(np.array(box, np.float32)).float()
        return image_data, box, disparity

    def get_random_data(self, index, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):

        line = self.train_lines[index].split()

        depth_path = os.path.join("/content/drive/MyDrive/cityscapes-to-coco-conversion-master/data/cityscapes/",
                                  line[0].replace("leftImg8bit", "disparity"))

        image = Image.open(
            "/content/drive/MyDrive/cityscapes-to-coco-conversion-master/data/cityscapes/"+line[0])
        image = cvtColor(image)

        iw, ih = image.size
        w, h = input_shape

        # box  = np.array(self.data["annotations"][index]['bbox'])
        box = np.array([np.array(list(map(int, box.split(','))))
                       for box in line[1:]])

        disparity = cv2.imread(
            depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        disparity = self.map_disparity(disparity)
        disparity = Image.fromarray(disparity)

        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            disparity = disparity.resize((nw, nh), resample=Image.NEAREST)
            new_depth = Image.new('F', (w, h), 0)
            new_depth.paste(disparity, (dx, dy))
            disparity_data = np.array(new_depth, np.float32)

            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                # discard invalid box
                box = box[np.logical_and(box_w > 1, box_h > 1)]

            return image_data, box, disparity_data

        new_ar = iw / ih * rand(1 - jitter, 1 + jitter) / \
            rand(1 - jitter, 1 + jitter)
        scale = rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        disparity = disparity.resize((nw, nh), resample=Image.NEAREST)

        dx = int(rand(0, w - nw))
        dy = int(rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        new_depth = Image.new('F', (w, h), 0)
        new_depth.paste(disparity, (dx, dy))
        disparity = new_depth

        flip = rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            disparity = disparity.transpose(Image.FLIP_LEFT_RIGHT)

        image_data = np.array(image, np.uint8)
        disparity_data = np.array(disparity, np.float32())

        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1

        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge(
            (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip:
                box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]

        return image_data, box, disparity_data

    def iou(self, box):

        inter_upleft = np.maximum(self.anchors[:, :2], box[:2])
        inter_botright = np.minimum(self.anchors[:, 2:4], box[2:])

        inter_wh = inter_botright - inter_upleft
        inter_wh = np.maximum(inter_wh, 0)
        inter = inter_wh[:, 0] * inter_wh[:, 1]

        area_true = (box[2] - box[0]) * (box[3] - box[1])

        area_gt = (self.anchors[:, 2] - self.anchors[:, 0]) * \
            (self.anchors[:, 3] - self.anchors[:, 1])

        union = area_true + area_gt - inter

        iou = inter / union
        return iou

    def encode_box(self, box, return_iou=True, variances=[0.1, 0.1, 0.2, 0.2]):

        iou = self.iou(box)
        encoded_box = np.zeros((self.num_anchors, 4 + return_iou))

        assign_mask = iou > self.overlap_threshold

        if not assign_mask.any():
            assign_mask[iou.argmax()] = True

        if return_iou:
            encoded_box[:, -1][assign_mask] = iou[assign_mask]

        assigned_anchors = self.anchors[assign_mask]

        box_center = 0.5 * (box[:2] + box[2:])
        box_wh = box[2:] - box[:2]

        assigned_anchors_center = (
            assigned_anchors[:, 0:2] + assigned_anchors[:, 2:4]) * 0.5
        assigned_anchors_wh = (
            assigned_anchors[:, 2:4] - assigned_anchors[:, 0:2])

        encoded_box[:, :2][assign_mask] = box_center - assigned_anchors_center
        encoded_box[:, :2][assign_mask] /= assigned_anchors_wh
        encoded_box[:, :2][assign_mask] /= np.array(variances)[:2]

        encoded_box[:, 2:4][assign_mask] = np.log(box_wh / assigned_anchors_wh)
        encoded_box[:, 2:4][assign_mask] /= np.array(variances)[2:4]
        return encoded_box.ravel()

    """
    Encode the output of the ssd based on the ground truth label and anchors
    """

    def assign_boxes(self, boxes):

        assignment = np.zeros((self.num_anchors, 4 + self.num_classes + 1))
        assignment[:, 4] = 1.0
        if len(boxes) == 0:
            return assignment

        encoded_boxes = np.apply_along_axis(self.encode_box, 1, boxes[:, :4])

        encoded_boxes = encoded_boxes.reshape(-1, self.num_anchors, 5)

        best_iou = encoded_boxes[:, :, -1].max(axis=0)
        best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
        best_iou_mask = best_iou > 0
        best_iou_idx = best_iou_idx[best_iou_mask]

        assign_num = len(best_iou_idx)

        encoded_boxes = encoded_boxes[:, best_iou_mask, :]

        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx,
                                                         np.arange(assign_num), :4]

        assignment[:, 4][best_iou_mask] = 0
        assignment[:, 5:-1][best_iou_mask] = boxes[best_iou_idx, 4:]

        assignment[:, -1][best_iou_mask] = 1

        return assignment

    def map_disparity(self, disparity):
        # https://github.com/mcordts/cityscapesScripts/issues/55#issuecomment-411486510
        # remap invalid points to -1 (not to conflict with 0, infinite depth, such as sky)
        disparity[disparity == 0] = -1
        # reduce by a factor of 4 based on the rescaled resolution
        disparity[disparity > -1] = (disparity[disparity > -1] - 1) / (256 * 4)
        disparity[disparity == -1] = 0
        return disparity


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a
