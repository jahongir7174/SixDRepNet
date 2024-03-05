import math
import random

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


def strip_optimizer(filename):
    x = torch.load(filename, map_location=torch.device('cpu'))
    x['model'].half()  # to FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, filename)


def resample():
    return random.choice((Image.BILINEAR, Image.BICUBIC))


def load_weight(model, ckpt):
    dst = model.state_dict()
    src = torch.load(ckpt, 'cpu')['model'].float().state_dict()
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


def weight_decay(model, decay):
    p1 = []
    p2 = []
    norm = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)

    for m in model.modules():
        for k, v in m.named_parameters(recurse=0):
            if k == "bias":  # bias (no decay)
                p1.append(v)
            elif k == "weight" and isinstance(m, norm):  # norm weight (no decay)
                p1.append(v)
            else:
                p2.append(v)  # weight (with decay)
    return [{'params': p1, 'weight_decay': 0.00},
            {'params': p2, 'weight_decay': decay}]


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
                 brightness: float = 0.4,
                 saturation: float = 0.4,
                 contrast: float = 0.4):
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
