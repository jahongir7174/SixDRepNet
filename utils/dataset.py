import random
from os.path import join, exists

import cv2
import numpy
import torch
from PIL import Image
from scipy.io import loadmat
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self,
                 dara_dir,
                 filepath,
                 transform, train=True):
        self.train = train
        self.data_dir = dara_dir
        self.transform = transform

        self.samples = self.load_label(filepath)

    def __getitem__(self, index):
        filename, kpt, pose = self.samples[index]
        image = self.load_image(join(self.data_dir, filename + '.jpg'))

        x1 = min(kpt[0, :])
        y1 = min(kpt[1, :])
        x2 = max(kpt[0, :])
        y2 = max(kpt[1, :])

        if self.train:
            k = numpy.random.random_sample() * 0.2 + 0.2
            x1 -= 0.6 * k * abs(x2 - x1)
            y1 -= 2 * k * abs(y2 - y1)
            x2 += 0.6 * k * abs(x2 - x1)
            y2 += 0.6 * k * abs(y2 - y1)
        else:
            k = 0.20
            x1 -= 2 * k * abs(x2 - x1)
            y1 -= 2 * k * abs(y2 - y1)
            x2 += 2 * k * abs(x2 - x1)
            y2 += 0.6 * k * abs(y2 - y1)

        image = image.crop((int(x1), int(y1), int(x2), int(y2)))

        p = pose[0]
        y = pose[1]
        r = pose[2]

        # horizontal flip
        if self.train and random.random() < 0.5:
            y = -y
            r = -r
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image = self.transform(image)

        if self.train:
            p = numpy.array([[1, 0, 0],
                             [0, numpy.cos(p), -numpy.sin(p)],
                             [0, numpy.sin(p), numpy.cos(p)]])
            y = numpy.array([[numpy.cos(y), 0, numpy.sin(y)],
                             [0, 1, 0],
                             [-numpy.sin(y), 0, numpy.cos(y)]])
            r = numpy.array([[numpy.cos(r), -numpy.sin(r), 0],
                             [numpy.sin(r), numpy.cos(r), 0],
                             [0, 0, 1]])
            return image, torch.FloatTensor(r.dot(y.dot(p)))
        else:
            return image, torch.FloatTensor([p, y, r])

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def load_image(filename):
        with open(filename, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
        return image

    def load_label(self, filepath):
        if exists(filepath.replace('txt', 'cache')):
            return torch.load(filepath.replace('txt', 'cache'))

        with open(filepath, 'r') as f:
            filenames = f.read().splitlines()

        samples = []
        for filename in filenames:
            label = loadmat(join(self.data_dir, filename + '.mat'))
            samples.append([filename, label['pt2d'], label['Pose_Para'][0][:3]])

        torch.save(samples, filepath.replace('txt', 'cache'))
        return samples


def augment_hsv(image, params):
    # HSV color-space augmentation
    h = params['hsv_h']
    s = params['hsv_s']
    v = params['hsv_v']
    image = numpy.array(image)

    r = numpy.random.uniform(-1, 1, 3) * [h, s, v] + 1

    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))

    x = numpy.arange(0, 256, dtype=r.dtype)
    lut_h = ((x * r[0]) % 180).astype('uint8')
    lut_s = numpy.clip(x * r[1], 0, 255).astype('uint8')
    lut_v = numpy.clip(x * r[2], 0, 255).astype('uint8')

    hsv = cv2.merge((cv2.LUT(h, lut_h), cv2.LUT(s, lut_s), cv2.LUT(v, lut_v)))
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return Image.fromarray(image)
