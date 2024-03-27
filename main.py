import argparse
import copy
import csv
import os
import warnings

import cv2
import numpy
import torch
import tqdm
from timm import utils
from torch.utils import data
from torchvision import transforms

from nets import nn
from utils import util
from utils.dataset import Dataset

warnings.filterwarnings("ignore")

data_dir = '../Dataset/POSE'
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def lr(args):
    return 5E-5 * args.batch_size * args.world_size / 64


def train(args):
    # Model
    model = nn.rep_net_a2()
    model = util.load_weights(model, ckpt='./weights/A2.pt')
    model.cuda()

    # Optimizer
    optimizer = torch.optim.Adam(util.params(model, lr(args)), lr(args))

    # EMA
    ema = nn.EMA(model) if args.local_rank == 0 else None

    sampler = None
    dataset = Dataset(f'{data_dir}/300W_LP',
                      f'{data_dir}/300W_LP/train.txt',
                      transforms.Compose([util.Resize(size=args.input_size),
                                          transforms.ToTensor(),
                                          normalize]),
                      train=True)

    if args.distributed:
        sampler = data.distributed.DistributedSampler(dataset)

    loader = data.DataLoader(dataset, args.batch_size, sampler is None,
                             sampler=sampler, num_workers=8, pin_memory=True)

    if args.distributed:
        # DDP mode
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(module=model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)

    best = float('inf')
    num_steps = len(loader)

    criterion = nn.ComputeLoss().cuda()
    amp_scale = torch.cuda.amp.GradScaler()
    scheduler = nn.CosineLR(args, optimizer)

    with open('weights/step.csv', 'w') as log:
        if args.local_rank == 0:
            logger = csv.DictWriter(log, fieldnames=['epoch', 'loss', 'Pitch', 'Yaw', 'Roll'])
            logger.writeheader()

        for epoch in range(args.epochs):
            model.train()

            if args.distributed:
                sampler.set_epoch(epoch)

            p_bar = loader
            avg_loss = util.AverageMeter()

            if args.local_rank == 0:
                print(('\n' + '%10s' * 3) % ('epoch', 'memory', 'loss'))
                p_bar = tqdm.tqdm(iterable=p_bar, total=num_steps)

            for samples, targets in p_bar:

                samples = samples.cuda()
                targets = targets.cuda()

                # Forward
                with torch.cuda.amp.autocast():
                    outputs = model(samples)
                loss = criterion(outputs, targets)

                optimizer.zero_grad()

                # Backward
                amp_scale.scale(loss).backward()

                # Optimize
                amp_scale.step(optimizer)
                amp_scale.update(None)
                if ema:
                    ema.update(model)

                # Log
                if args.distributed:
                    loss = utils.reduce_tensor(loss.data, args.world_size)
                avg_loss.update(loss.item(), samples.size(0))
                if args.local_rank == 0:
                    memory = f'{torch.cuda.memory_reserved() / 1E9:.3g}G'
                    s = ('%10s' * 2 + '%10.3g') % (f'{epoch + 1}/{args.epochs}', memory, avg_loss.avg)
                    p_bar.set_description(s)

            scheduler.step(epoch, optimizer)

            if args.local_rank == 0:
                last = test(args, ema.ema)

                logger.writerow({'Pitch': str(f'{last[0]:.3f}'),
                                 'Yaw': str(f'{last[1]:.3f}'),
                                 'Roll': str(f'{last[2]:.3f}'),
                                 'loss': str(f'{avg_loss.avg:.3f}'),
                                 'epoch': str(epoch + 1).zfill(3)})
                log.flush()

                # Update best MAE
                if best > sum(last):
                    best = sum(last)

                # Save model
                save = {'epoch': epoch,
                        'model': copy.deepcopy(ema.ema).half()}

                # Save last, best and delete
                torch.save(save, f='./weights/last.pt')
                if best == sum(last):
                    torch.save(save, f='./weights/best.pt')
                del save

    if args.local_rank == 0:
        util.strip_optimizer('./weights/best.pt')  # strip optimizers
        util.strip_optimizer('./weights/last.pt')  # strip optimizers

    torch.cuda.empty_cache()


@torch.no_grad()
def test(args, model=None):
    dataset = Dataset(f'{data_dir}/AFLW2000',
                      f'{data_dir}/AFLW2000/test.txt',
                      transforms.Compose([transforms.Resize(args.input_size + 32),
                                          transforms.CenterCrop(args.input_size),
                                          transforms.ToTensor(),
                                          normalize]),
                      train=False)
    loader = data.DataLoader(dataset, batch_size=2)

    if model is None:
        model = torch.load(f=f'./weights/best.pt', map_location='cuda')
        model = model['model'].float().fuse()

    model.eval()

    total = 0
    y_error = 0
    p_error = 0
    r_error = 0
    for sample, target in tqdm.tqdm(loader, ('%10s' * 3) % ('Pitch', 'Yaw', 'Roll')):
        sample = sample.cuda()
        total += target.size(0)

        p_target = target[:, 0].float() * 180 / numpy.pi
        y_target = target[:, 1].float() * 180 / numpy.pi
        r_target = target[:, 2].float() * 180 / numpy.pi

        output = model(sample)
        output = util.compute_euler(output) * 180 / numpy.pi

        p_output = output[:, 0].cpu()
        y_output = output[:, 1].cpu()
        r_output = output[:, 2].cpu()

        p_error += torch.sum(torch.min(torch.stack((torch.abs(p_target - p_output),
                                                    torch.abs(p_output + 360 - p_target),
                                                    torch.abs(p_output - 360 - p_target),
                                                    torch.abs(p_output + 180 - p_target),
                                                    torch.abs(p_output - 180 - p_target))), 0)[0])
        y_error += torch.sum(torch.min(torch.stack((torch.abs(y_target - y_output),
                                                    torch.abs(y_output + 360 - y_target),
                                                    torch.abs(y_output - 360 - y_target),
                                                    torch.abs(y_output + 180 - y_target),
                                                    torch.abs(y_output - 180 - y_target))), 0)[0])
        r_error += torch.sum(torch.min(torch.stack((torch.abs(r_target - r_output),
                                                    torch.abs(r_output + 360 - r_target),
                                                    torch.abs(r_output - 360 - r_target),
                                                    torch.abs(r_output + 180 - r_target),
                                                    torch.abs(r_output - 180 - r_target))), 0)[0])
    # Print results
    p_error, y_error, r_error = p_error / total, y_error / total, r_error / total
    print(('%10s' * 3) % (f'{p_error:.3f}', f'{y_error:.3f}', f'{r_error:.3f}'))

    # Return results
    model.float()  # for training
    return p_error, y_error, r_error


@torch.no_grad()
def demo(args):
    from PIL import Image
    transform = transforms.Compose([transforms.Resize(args.input_size + 32),
                                    transforms.CenterCrop(args.input_size),
                                    transforms.ToTensor(),
                                    normalize])

    model = torch.load(f='./weights/best.pt', map_location='cuda')
    model = model['model'].float().fuse()

    detector = util.FaceDetector('./weights/detection.onnx')

    model.half()
    model.eval()

    stream = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if not stream.isOpened():
        print("Error opening video stream or file")

    # Read until video is completed
    while stream.isOpened():
        # Capture frame-by-frame
        success, frame = stream.read()
        if success:
            boxes = detector.detect(frame, (640, 640))
            boxes = boxes.astype('int32')
            for box in boxes:
                x_min = box[0]
                y_min = box[1]
                x_max = box[2]
                y_max = box[3]
                box_w = abs(x_max - x_min)
                box_h = abs(y_max - y_min)

                x_min = max(0, x_min - int(0.2 * box_h))
                y_min = max(0, y_min - int(0.2 * box_w))
                x_max = x_max + int(0.2 * box_h)
                y_max = y_max + int(0.2 * box_w)

                image = frame[y_min:y_max, x_min:x_max, :]
                image = Image.fromarray(image)
                image = image.convert('RGB')
                image = transform(image)
                image = image.unsqueeze(0)

                image = image.cuda()
                image = image.half()

                output = model(image)
                output = util.compute_euler(output) * 180 / numpy.pi

                p_output = output[:, 0].cpu()
                y_output = output[:, 1].cpu()
                r_output = output[:, 2].cpu()
                util.plot_pose_cube(frame,
                                    y_output, p_output, r_output,
                                    x_min + int(0.5 * (x_max - x_min)),
                                    y_min + int(0.5 * (y_max - y_min)), box_w)

            cv2.imshow('Demo', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # Break the loop
        else:
            break
    # When everything done, release the video capture object
    stream.release()
    # Closes all the frames
    cv2.destroyAllWindows()


def profile(args):
    import thop
    model = nn.rep_net_a2().fuse()
    shape = (1, 3, args.input_size, args.input_size)

    model.eval()
    model(torch.zeros(shape))

    x = torch.empty(shape)
    flops, num_params = thop.profile(copy.copy(model), inputs=[x], verbose=False)
    flops, num_params = thop.clever_format(nums=[flops, num_params], format="%.3f")

    if args.local_rank == 0:
        print(f'Number of parameters: {num_params}')
        print(f'Number of FLOPs: {flops}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', default=224, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--demo', action='store_true')

    args = parser.parse_args()

    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1

    if args.distributed:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.local_rank == 0:
        if not os.path.exists('weights'):
            os.makedirs('weights')

    util.setup_seed()
    util.setup_multi_processes()

    profile(args)

    if args.train:
        train(args)
    if args.test:
        test(args)
    if args.demo:
        demo(args)


if __name__ == "__main__":
    main()
