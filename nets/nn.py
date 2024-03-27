import copy
import math

import numpy
import torch


def normalize(v):
    mag = torch.sqrt(torch.sum(v.pow(2), dim=1, keepdim=True))
    eps = torch.FloatTensor([1E-8]).to(mag.device)
    mag = torch.max(mag, eps)
    return v / mag


def cross_product(u, v):
    shape = u.shape

    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    i = i.view(shape[0], 1)
    j = j.view(shape[0], 1)
    k = k.view(shape[0], 1)

    return torch.cat(tensors=(i, j, k), dim=1)


class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=0):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.norm = torch.nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.norm(self.conv(x))


class Residual(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k, s, p):
        super().__init__()

        assert k == 3
        assert p == 1
        self.in_channels = in_ch

        self.relu = torch.nn.ReLU()
        self.conv = torch.nn.Identity()

        self.conv1 = Conv(in_ch, out_ch, k=k, s=s, p=p)
        self.conv2 = Conv(in_ch, out_ch, k=1, s=s, p=p - k // 2)
        self.identity = torch.nn.BatchNorm2d(in_ch) if in_ch == out_ch and s == 1 else None

    @staticmethod
    def __pad(k):
        if k is None:
            return 0
        else:
            return torch.nn.functional.pad(k, pad=[1, 1, 1, 1])

    def __fuse_norm(self, m):
        if m is None:
            return 0, 0
        if isinstance(m, Conv):
            kernel = m.conv.weight
            running_mean = m.norm.running_mean
            running_var = m.norm.running_var
            gamma = m.norm.weight
            beta = m.norm.bias
            eps = m.norm.eps
        else:
            assert isinstance(m, torch.nn.BatchNorm2d)
            if not hasattr(self, 'norm'):
                in_channels = self.conv1.conv.in_channels
                kernel_value = numpy.zeros((in_channels, in_channels, 3, 3), dtype=numpy.float32)
                for i in range(in_channels):
                    kernel_value[i, i % in_channels, 1, 1] = 1
                self.norm = torch.from_numpy(kernel_value).to(m.weight.device)
            kernel = self.norm
            running_mean = m.running_mean
            running_var = m.running_var
            gamma = m.weight
            beta = m.bias
            eps = m.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def forward(self, x):
        if self.identity is None:
            return self.relu(self.conv1(x) + self.conv2(x))
        else:
            return self.relu(self.conv1(x) + self.conv2(x) + self.identity(x))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))

    def fuse(self):
        k1, b1 = self.__fuse_norm(self.conv1)
        k2, b2 = self.__fuse_norm(self.conv2)
        k3, b3 = self.__fuse_norm(self.identity)

        self.conv = torch.nn.Conv2d(in_channels=self.conv1.conv.in_channels,
                                    out_channels=self.conv1.conv.out_channels,
                                    kernel_size=self.conv1.conv.kernel_size,
                                    stride=self.conv1.conv.stride,
                                    padding=self.conv1.conv.padding,
                                    dilation=self.conv1.conv.dilation,
                                    groups=self.conv1.conv.groups, bias=True)

        self.conv.weight.data = k1 + self.__pad(k2) + k3
        self.conv.bias.data = b1 + b2 + b3

        if hasattr(self, 'conv1'):
            self.__delattr__('conv1')
        if hasattr(self, 'conv2'):
            self.__delattr__('conv2')
        if hasattr(self, 'identity'):
            self.__delattr__('identity')
        if hasattr(self, 'norm'):
            self.__delattr__('norm')
        self.forward = self.fuse_forward


class SixDRepVGG(torch.nn.Module):
    def __init__(self, width, depth, num_classes=6):
        super().__init__()

        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.p4 = []
        self.p5 = []

        # p1
        self.p1.append(Residual(width[0], width[1], k=3, s=2, p=1))
        # p2
        for i in range(depth[0]):
            if i == 0:
                self.p2.append(Residual(width[1], width[2], k=3, s=2, p=1))
            else:
                self.p2.append(Residual(width[2], width[2], k=3, s=1, p=1))
        # p3
        for i in range(depth[1]):
            if i == 0:
                self.p3.append(Residual(width[2], width[3], k=3, s=2, p=1))
            else:
                self.p3.append(Residual(width[3], width[3], k=3, s=1, p=1))
        # p4
        for i in range(depth[2]):
            if i == 0:
                self.p4.append(Residual(width[3], width[4], k=3, s=2, p=1))
            else:
                self.p4.append(Residual(width[4], width[4], k=3, s=1, p=1))
        # p5
        for i in range(depth[3]):
            if i == 0:
                self.p5.append(Residual(width[4], width[5], k=3, s=2, p=1))
            else:
                self.p5.append(Residual(width[5], width[5], k=3, s=1, p=1))

        self.p1 = torch.nn.Sequential(*self.p1)
        self.p2 = torch.nn.Sequential(*self.p2)
        self.p3 = torch.nn.Sequential(*self.p3)
        self.p4 = torch.nn.Sequential(*self.p4)
        self.p5 = torch.nn.Sequential(*self.p5)
        self.fc = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),
                                      torch.nn.Flatten(),
                                      torch.nn.Linear(width[5], num_classes))

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        fc = self.fc(p5)

        x_raw = fc[:, 0:3]
        y_raw = fc[:, 3:6]

        x = normalize(x_raw)
        z = cross_product(x, y_raw)
        z = normalize(z)
        y = cross_product(z, x)

        x = x.view(-1, 3, 1)
        y = y.view(-1, 3, 1)
        z = z.view(-1, 3, 1)
        return torch.cat(tensors=(x, y, z), dim=2)

    def fuse(self):
        for m in self.modules():
            if type(m) is Residual:
                m.fuse()
        return self


def rep_net_a0():
    return SixDRepVGG(width=(3, 48, 48, 96, 192, 1280), depth=(2, 4, 14, 1))


def rep_net_a1():
    return SixDRepVGG(width=(3, 64, 64, 128, 256, 1280), depth=[2, 4, 14, 1])


def rep_net_a2():
    return SixDRepVGG(width=[3, 64, 96, 192, 384, 1408], depth=[2, 4, 14, 1])


def rep_net_b0():
    return SixDRepVGG(width=[3, 64, 64, 128, 256, 1280], depth=[4, 6, 16, 1])


def rep_net_b1():
    return SixDRepVGG(width=[3, 64, 128, 256, 512, 2048], depth=[4, 6, 16, 1])


def rep_net_b2():
    return SixDRepVGG(width=[3, 64, 160, 320, 640, 2560], depth=[4, 6, 16, 1])


class EMA:
    """
    Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = copy.deepcopy(model).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        if hasattr(model, 'module'):
            model = model.module
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()


class CosineLR:
    def __init__(self, args, optimizer):
        self.min_lr = 1E-6
        self.epochs = args.epochs
        self.learning_rates = [x['lr'] for x in optimizer.param_groups]

    def step(self, epoch, optimizer):
        param_groups = optimizer.param_groups
        for param_group, lr in zip(param_groups, self.learning_rates):
            alpha = math.cos(math.pi * epoch / self.epochs)
            lr = 0.5 * (lr - self.min_lr) * (1 + alpha)
            param_group['lr'] = self.min_lr + lr


class ComputeLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1E-7

    def forward(self, outputs, targets):
        m = torch.bmm(targets, outputs.transpose(1, 2))
        cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
        theta = torch.acos(torch.clamp(cos, -1 + self.eps, 1 - self.eps))

        return torch.mean(theta)
