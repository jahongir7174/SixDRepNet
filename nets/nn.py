import math

import numpy
import torch


def normalize(v):
    mag = torch.sqrt(torch.sum(v.pow(2), dim=1, keepdim=True))
    eps = torch.FloatTensor([1e-8]).to(mag.device)
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
    def __init__(self, in_ch, out_ch, k=1, s=1, p=0, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
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
            return torch.nn.functional.pad(k, [1, 1, 1, 1])

    def __fuse_norm(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.norm.running_mean
            running_var = branch.norm.running_var
            gamma = branch.norm.weight
            beta = branch.norm.bias
            eps = branch.norm.eps
        else:
            assert isinstance(branch, torch.nn.BatchNorm2d)
            if not hasattr(self, 'norm'):
                input_dim = self.in_channels
                kernel_value = numpy.zeros((self.in_channels, input_dim, 3, 3), dtype=numpy.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.norm = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.norm
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
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
    def __init__(self, width, depth, num_classes=1000):
        super().__init__()
        assert len(width) == 4
        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.p4 = []
        self.p5 = []

        filters = [3, min(64, int(64 * width[0])), 64, 128, 256, 512]
        for i, (x, y) in enumerate(zip(filters[2:], width)):
            filters[i + 2] = int(x * y)

        # p1
        self.p1.append(Residual(filters[0], filters[1], k=3, s=2, p=1))
        # p2
        for i in range(depth[0]):
            if i == 0:
                self.p2.append(Residual(filters[1], filters[2], k=3, s=2, p=1))
            else:
                self.p2.append(Residual(filters[2], filters[2], k=3, s=1, p=1))
        # p3
        for i in range(depth[1]):
            if i == 0:
                self.p3.append(Residual(filters[2], filters[3], k=3, s=2, p=1))
            else:
                self.p3.append(Residual(filters[3], filters[3], k=3, s=1, p=1))
        # p4
        for i in range(depth[2]):
            if i == 0:
                self.p4.append(Residual(filters[3], filters[4], k=3, s=2, p=1))
            else:
                self.p4.append(Residual(filters[4], filters[4], k=3, s=1, p=1))
        # p5
        for i in range(depth[3]):
            if i == 0:
                self.p5.append(Residual(filters[4], filters[5], k=3, s=2, p=1))
            else:
                self.p5.append(Residual(filters[5], filters[5], k=3, s=1, p=1))

        self.p1 = torch.nn.Sequential(*self.p1)
        self.p2 = torch.nn.Sequential(*self.p2)
        self.p3 = torch.nn.Sequential(*self.p3)
        self.p4 = torch.nn.Sequential(*self.p4)
        self.p5 = torch.nn.Sequential(*self.p5)
        self.fc = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),
                                      torch.nn.Flatten(),
                                      torch.nn.Linear(filters[5], num_classes))

        self.init_weights()

    def forward(self, x):
        x = self.p1(x)
        x = self.p2(x)
        x = self.p3(x)
        x = self.p4(x)
        x = self.p5(x)
        x = self.fc(x)

        x_raw = x[:, 0:3]
        y_raw = x[:, 3:6]

        x = normalize(x_raw)
        z = cross_product(x, y_raw)
        z = normalize(z)
        y = cross_product(z, x)

        x = x.view(-1, 3, 1)
        y = y.view(-1, 3, 1)
        z = z.view(-1, 3, 1)
        return torch.cat(tensors=(x, y, z), dim=2)

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.BatchNorm2d:
                m.eps = 1E-3
                m.momentum = 0.03

    def fuse(self):
        for m in self.modules():
            if type(m) is Residual:
                m.fuse()
        return self


def rep_net_a0():
    return SixDRepVGG(width=[0.75, 0.75, 0.75, 2.5], depth=[2, 4, 14, 1])


def rep_net_a1():
    return SixDRepVGG(width=[1, 1, 1, 2.5], depth=[2, 4, 14, 1])


def rep_net_a2():
    return SixDRepVGG(width=[1.5, 1.5, 1.5, 2.75], depth=[2, 4, 14, 1])


def rep_net_b0():
    return SixDRepVGG(width=[1, 1, 1, 2.5], depth=[4, 6, 16, 1])


def rep_net_b1():
    return SixDRepVGG(width=[2, 2, 2, 4], depth=[4, 6, 16, 1])


def rep_net_b2():
    return SixDRepVGG(width=[2.5, 2.5, 2.5, 5], depth=[4, 6, 16, 1])


class EMA:
    """
    Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        import copy
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


class LinearLR:
    def __init__(self, args, params, num_steps):
        min_lr = params['min_lr']
        max_lr = params['max_lr']

        total_steps = args.epochs * num_steps
        warmup_steps = max(params['warmup_epochs'] * num_steps, 100)

        warmup_lr = numpy.linspace(min_lr, max_lr, int(warmup_steps), endpoint=False)
        decay_lr = numpy.linspace(max_lr, min_lr, int(total_steps - warmup_steps))

        self.total_lr = numpy.concatenate((warmup_lr, decay_lr))

    def step(self, step, optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.total_lr[step]


class RMSprop(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, alpha=0.9, eps=0.001, weight_decay=0.0,
                 momentum=0.9, centered=False, decoupled_decay=False, lr_in_momentum=True):

        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum,
                        centered=centered, decoupled_decay=decoupled_decay, lr_in_momentum=lr_in_momentum)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for param_group in self.param_groups:
            param_group.setdefault('momentum', 0)
            param_group.setdefault('centered', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for param_group in self.param_groups:
            for param in param_group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Optimizer does not support sparse gradients')
                state = self.state[param]
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.ones_like(param.data)
                    if param_group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(param.data)
                    if param_group['centered']:
                        state['grad_avg'] = torch.zeros_like(param.data)

                square_avg = state['square_avg']
                one_minus_alpha = 1. - param_group['alpha']

                state['step'] += 1

                if param_group['weight_decay'] != 0:
                    if 'decoupled_decay' in param_group and param_group['decoupled_decay']:
                        param.data.add_(param.data, alpha=-param_group['weight_decay'])
                    else:
                        grad = grad.add(param.data, alpha=param_group['weight_decay'])

                square_avg.add_(grad.pow(2) - square_avg, alpha=one_minus_alpha)

                if param_group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.add_(grad - grad_avg, alpha=one_minus_alpha)
                    avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).add(param_group['eps']).sqrt_()
                else:
                    avg = square_avg.add(param_group['eps']).sqrt_()

                if param_group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    if 'lr_in_momentum' in param_group and param_group['lr_in_momentum']:
                        buf.mul_(param_group['momentum']).addcdiv_(grad, avg, value=param_group['lr'])
                        param.data.add_(-buf)
                    else:
                        buf.mul_(param_group['momentum']).addcdiv_(grad, avg)
                        param.data.add_(-param_group['lr'], buf)
                else:
                    param.data.addcdiv_(grad, avg, value=-param_group['lr'])

        return loss


class ComputeLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1E-7

    def forward(self, outputs, targets):
        m = torch.bmm(targets, outputs.transpose(1, 2))
        cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
        theta = torch.acos(torch.clamp(cos, -1 + self.eps, 1 - self.eps))

        return torch.mean(theta)
