import math
import os
import random

import cv2
import numpy
import torch
from PIL import Image
from PIL import ImageEnhance


def setup_seed():
    """
    Setup random seed.
    """
    random.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_multi_processes():
    """
    Setup multi-processing environment variables.
    """
    import cv2
    from os import environ
    from platform import system

    # set multiprocess start method as `fork` to speed up the training
    if system() != 'Windows':
        torch.multiprocessing.set_start_method('fork', force=True)

    # disable opencv multithreading to avoid system being overloaded
    cv2.setNumThreads(0)

    # setup OMP threads
    if 'OMP_NUM_THREADS' not in environ:
        environ['OMP_NUM_THREADS'] = '1'

    # setup MKL threads
    if 'MKL_NUM_THREADS' not in environ:
        environ['MKL_NUM_THREADS'] = '1'


def plot_lr(args, optimizer, scheduler):
    import copy
    from matplotlib import pyplot

    optimizer = copy.copy(optimizer)
    scheduler = copy.copy(scheduler)

    y = []
    for epoch in range(args.epochs):
        y.append(optimizer.param_groups[-1]['lr'])
        scheduler.step(epoch + 1, optimizer)

    pyplot.plot(y, '.-', label='LR')
    pyplot.xlabel('epoch')
    pyplot.ylabel('LR')
    pyplot.grid()
    pyplot.xlim(0, args.epochs)
    pyplot.ylim(0)
    pyplot.savefig('./weights/lr.png', dpi=200)
    pyplot.close()


def strip_optimizer(filename):
    x = torch.load(filename, map_location=torch.device('cpu'))
    x['model'].half()  # to FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, filename)


def resample():
    return random.choice((Image.BILINEAR, Image.BICUBIC))


def load_weights(model, ckpt):
    dst = model.state_dict()
    src = torch.load(ckpt)['model']
    src = src.cpu().float().state_dict()

    ckpt = {}
    for k, v in src.items():
        if k in dst and v.shape == dst[k].shape:
            ckpt[k] = v
    model.load_state_dict(state_dict=ckpt, strict=False)
    return model


def compute_euler(matrices):
    shape = matrices.shape
    sy = matrices[:, 0, 0] * matrices[:, 0, 0] + matrices[:, 1, 0] * matrices[:, 1, 0]
    sy = torch.sqrt(sy)
    singular = (sy < 1E-6).float()

    x = torch.atan2(matrices[:, 2, 1], matrices[:, 2, 2])
    y = torch.atan2(-matrices[:, 2, 0], sy)
    z = torch.atan2(matrices[:, 1, 0], matrices[:, 0, 0])

    xs = torch.atan2(-matrices[:, 1, 2], matrices[:, 1, 1])
    ys = torch.atan2(-matrices[:, 2, 0], sy)
    zs = torch.zeros_like(z)

    device = matrices.device
    out_euler = torch.zeros(shape[0], 3, device=device)
    out_euler[:, 0] = x * (1 - singular) + xs * singular
    out_euler[:, 1] = y * (1 - singular) + ys * singular
    out_euler[:, 2] = z * (1 - singular) + zs * singular
    return out_euler


def params(model, lr):
    return [{'params': model.p1.parameters(), 'lr': lr},
            {'params': model.p2.parameters(), 'lr': lr},
            {'params': model.p3.parameters(), 'lr': lr},
            {'params': model.p4.parameters(), 'lr': lr},
            {'params': model.p5.parameters(), 'lr': lr},
            {'params': model.fc.parameters(), 'lr': lr * 10}]


class Resize:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, image):
        size = self.size
        i, j, h, w = self.params(image.size)
        image = image.crop((j, i, j + w, i + h))
        return image.resize([size, size], resample())

    @staticmethod
    def params(size):
        scale = (0.8, 1.0)
        ratio = (3. / 4., 4. / 3.)
        for _ in range(10):
            target_area = random.uniform(*scale) * size[0] * size[1]
            aspect_ratio = math.exp(random.uniform(*(math.log(ratio[0]), math.log(ratio[1]))))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= size[0] and h <= size[1]:
                i = random.randint(0, size[1] - h)
                j = random.randint(0, size[0] - w)
                return i, j, h, w

        if (size[0] / size[1]) < min(ratio):
            w = size[0]
            h = int(round(w / min(ratio)))
        elif (size[0] / size[1]) > max(ratio):
            h = size[1]
            w = int(round(h * max(ratio)))
        else:
            w = size[0]
            h = size[1]
        i = (size[1] - h) // 2
        j = (size[0] - w) // 2
        return i, j, h, w


class ColorJitter:
    def __init__(self,
                 p: float = 0.5,
                 brightness: float = 0.1,
                 saturation: float = 0.1,
                 contrast: float = 0.1):
        self.brightness = (1 - brightness, 1 + brightness)
        self.saturation = (1 - saturation, 1 + saturation)
        self.contrast = (1 - contrast, 1 + contrast)
        self.indices = [0, 1, 2]
        self.p = p

    def __call__(self, image):
        if random.random() > self.p:
            return image

        b = random.uniform(self.brightness[0], self.brightness[1])
        s = random.uniform(self.saturation[0], self.saturation[1])
        c = random.uniform(self.contrast[0], self.contrast[1])

        random.shuffle(self.indices)

        for i in self.indices:
            if i == 0:
                image = ImageEnhance.Brightness(image).enhance(b)  # brightness
            elif i == 1:
                image = ImageEnhance.Contrast(image).enhance(c)  # contrast
            elif i == 2:
                image = ImageEnhance.Color(image).enhance(s)  # saturation

        return image


class AverageMeter:
    def __init__(self):
        self.num = 0
        self.sum = 0
        self.avg = 0

    def update(self, v, n):
        self.num = self.num + n
        self.sum = self.sum + v * n
        self.avg = self.sum / self.num


def plot_pose_cube(image, yaw, pitch, roll, tdx=None, tdy=None, size=150.):
    p = pitch * numpy.pi / 180
    y = -(yaw * numpy.pi / 180)
    r = roll * numpy.pi / 180
    if (tdx is not None) and (tdy is not None):
        face_x = tdx - 0.50 * size
        face_y = tdy - 0.50 * size
    else:
        height, width = image.shape[:2]
        face_x = width / 2 - 0.5 * size
        face_y = height / 2 - 0.5 * size

    x1 = size * (math.cos(y) * math.cos(r)) + face_x
    y1 = size * (math.cos(p) * math.sin(r) + math.cos(r) * math.sin(p) * math.sin(y)) + face_y
    x2 = size * (-math.cos(y) * math.sin(r)) + face_x
    y2 = size * (math.cos(p) * math.cos(r) - math.sin(p) * math.sin(y) * math.sin(r)) + face_y
    x3 = size * (math.sin(y)) + face_x
    y3 = size * (-math.cos(y) * math.sin(p)) + face_y

    # Draw base in red
    cv2.line(image, (int(face_x), int(face_y)), (int(x1), int(y1)), (0, 0, 255), 3)
    cv2.line(image, (int(face_x), int(face_y)), (int(x2), int(y2)), (0, 0, 255), 3)
    cv2.line(image, (int(x2), int(y2)), (int(x2 + x1 - face_x), int(y2 + y1 - face_y)), (0, 0, 255), 3)
    cv2.line(image, (int(x1), int(y1)), (int(x1 + x2 - face_x), int(y1 + y2 - face_y)), (0, 0, 255), 3)
    # Draw pillars in blue
    cv2.line(image, (int(face_x), int(face_y)), (int(x3), int(y3)), (255, 0, 0), 2)
    cv2.line(image, (int(x1), int(y1)), (int(x1 + x3 - face_x), int(y1 + y3 - face_y)), (255, 0, 0), 2)
    cv2.line(image, (int(x2), int(y2)), (int(x2 + x3 - face_x), int(y2 + y3 - face_y)), (255, 0, 0), 2)
    cv2.line(image, (int(x2 + x1 - face_x), int(y2 + y1 - face_y)),
             (int(x3 + x1 + x2 - 2 * face_x), int(y3 + y2 + y1 - 2 * face_y)), (255, 0, 0), 2)
    # Draw top in green
    cv2.line(image, (int(x3 + x1 - face_x), int(y3 + y1 - face_y)),
             (int(x3 + x1 + x2 - 2 * face_x), int(y3 + y2 + y1 - 2 * face_y)), (0, 255, 0), 2)
    cv2.line(image, (int(x2 + x3 - face_x), int(y2 + y3 - face_y)),
             (int(x3 + x1 + x2 - 2 * face_x), int(y3 + y2 + y1 - 2 * face_y)), (0, 255, 0), 2)
    cv2.line(image, (int(x3), int(y3)), (int(x3 + x1 - face_x), int(y3 + y1 - face_y)), (0, 255, 0), 2)
    cv2.line(image, (int(x3), int(y3)), (int(x3 + x2 - face_x), int(y3 + y2 - face_y)), (0, 255, 0), 2)

    return image


def distance2box(points, distance, max_shape=None):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return numpy.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    outputs = []
    for i in range(0, distance.shape[1], 2):
        p_x = points[:, i % 2] + distance[:, i]
        p_y = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            p_x = p_x.clamp(min=0, max=max_shape[1])
            p_y = p_y.clamp(min=0, max=max_shape[0])
        outputs.append(p_x)
        outputs.append(p_y)
    return numpy.stack(outputs, axis=-1)


class FaceDetector:
    def __init__(self, onnx_path=None, session=None):
        from onnxruntime import InferenceSession
        self.session = session

        self.batched = False
        if self.session is None:
            assert onnx_path is not None
            assert os.path.exists(onnx_path)
            self.session = InferenceSession(onnx_path,
                                            providers=['CUDAExecutionProvider'])
        self.nms_thresh = 0.4
        self.center_cache = {}
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        if isinstance(input_shape[2], str):
            self.input_size = None
        else:
            self.input_size = tuple(input_shape[2:4][::-1])
        input_name = input_cfg.name
        outputs = self.session.get_outputs()
        if len(outputs[0].shape) == 3:
            self.batched = True
        output_names = []
        for output in outputs:
            output_names.append(output.name)
        self.input_name = input_name
        self.output_names = output_names
        self.use_kps = False
        self._num_anchors = 1
        if len(outputs) == 6:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif len(outputs) == 9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        elif len(outputs) == 10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif len(outputs) == 15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self.use_kps = True

    def forward(self, x, score_thresh):
        scores_list = []
        bboxes_list = []
        points_list = []
        input_size = tuple(x.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(x,
                                     1.0 / 128,
                                     input_size,
                                     (127.5, 127.5, 127.5), swapRB=True)
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        input_height = blob.shape[2]
        input_width = blob.shape[3]
        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            if self.batched:
                scores = outputs[idx][0]
                boxes = outputs[idx + fmc][0]
                boxes = boxes * stride
            else:
                scores = outputs[idx]
                boxes = outputs[idx + fmc]
                boxes = boxes * stride

            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = numpy.stack(numpy.mgrid[:height, :width][::-1], axis=-1)
                anchor_centers = anchor_centers.astype(numpy.float32)

                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = numpy.stack([anchor_centers] * self._num_anchors, axis=1)
                    anchor_centers = anchor_centers.reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_indices = numpy.where(scores >= score_thresh)[0]
            bboxes = distance2box(anchor_centers, boxes)
            pos_scores = scores[pos_indices]
            pos_bboxes = bboxes[pos_indices]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
        return scores_list, bboxes_list

    def detect(self, image, input_size=None, score_threshold=0.5, max_num=0, metric='default'):
        assert input_size is not None or self.input_size is not None
        input_size = self.input_size if input_size is None else input_size
        image_ratio = float(image.shape[0]) / image.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if image_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / image_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * image_ratio)
        det_scale = float(new_height) / image.shape[0]
        resized_img = cv2.resize(image, (new_width, new_height))
        det_img = numpy.zeros((input_size[1], input_size[0], 3), dtype=numpy.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        scores_list, bboxes_list = self.forward(det_img, score_threshold)

        scores = numpy.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = numpy.vstack(bboxes_list) / det_scale
        pre_det = numpy.hstack((bboxes, scores)).astype(numpy.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        if 0 < max_num < det.shape[0]:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = image.shape[0] // 2, image.shape[1] // 2
            offsets = numpy.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                    (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
            offset_dist_squared = numpy.sum(numpy.power(offsets, 2.0), 0)
            if metric == 'max':
                values = area
            else:
                values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            index = numpy.argsort(values)[::-1]  # some extra weight on the centering
            index = index[0:max_num]
            det = det[index, :]
        return det

    def nms(self, outputs):
        thresh = self.nms_thresh
        x1 = outputs[:, 0]
        y1 = outputs[:, 1]
        x2 = outputs[:, 2]
        y2 = outputs[:, 3]
        scores = outputs[:, 4]

        order = scores.argsort()[::-1]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = numpy.maximum(x1[i], x1[order[1:]])
            yy1 = numpy.maximum(y1[i], y1[order[1:]])
            xx2 = numpy.minimum(x2[i], x2[order[1:]])
            yy2 = numpy.minimum(y2[i], y2[order[1:]])

            w = numpy.maximum(0.0, xx2 - xx1 + 1)
            h = numpy.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            indices = numpy.where(ovr <= thresh)[0]
            order = order[indices + 1]

        return keep
