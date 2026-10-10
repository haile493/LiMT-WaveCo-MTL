import argparse
import logging
import os
import random
import shutil
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from segmentation_models_pytorch as smp
from torchvision import transforms
from tqdm import tqdm
from dataloaders.brats_LiTS import (BraTS, CenterCrop, RandomCrop,
                                   RandomRotFlip, ToTensor,
                                   TwoStreamBatchSampler)
from networks.net_factory_3d import net_factory_3d
from utils import losses, metrics, ramps
from val_LiTS import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data_LiTS_sampled', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='pretrain_LiTS', help='experiment_name')
parser.add_argument('--model', type=str,
                    default=' Unet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[96, 96, 96],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--labeled_num', type=int, default=105,
                    help='labeled data')

args = parser.parse_args()


def train(args, snapshot_path):
    base_lr = args.base_lr
    train_data_path = args.root_path
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    num_classes = 3
    model = net_factory_3d(net_type=args.model, in_chns=1, class_num=num_classes)
    db_train = BraTS(base_dir=train_data_path,
                         split='train',
                         num=args.labeled_num,
                         transform=transforms.Compose([
                             RandomCrop(args.patch_size),
                             ToTensor(),
                         ]))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(3)
    focal_loss = smp.losses.FocalLoss(mode='multiclass')

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_organ_performance = 0.0
    best_tumor_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)

            loss_ce = ce_loss(outputs, label_batch)
            loss_dice = dice_loss(outputs_soft, label_batch.unsqueeze(1))
            loss_focal = focal_loss(outputs,  label_batch[:].long())

            loss = loss_dice + loss_ce +loss_focal
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))
            writer.add_scalar('loss/loss', loss, iter_num)

            if iter_num > 0 and iter_num % 400 == 0:
                model.eval()
                avg_metric = test_all_case(
                    model, args.root_path, test_list="test.txt", num_classes=3, patch_size=args.patch_size,
                    stride_xy=32, stride_z=16)
                if avg_metric[0, 0] > best_organ_performance or avg_metric[1, 0] > best_tumor_performance:
                    if avg_metric[0, 0] > best_organ_performance:
                        best_organ_performance = avg_metric[0, 0]
                    if avg_metric[1, 0] > best_tumor_performance:
                        best_tumor_performance = avg_metric[1, 0]
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_organ_dice_{}_tumor_dice_{}.pth'.format(
                                                      iter_num, round(avg_metric[0, 0], 4),
                                                      round(avg_metric[1, 0], 4)))
                    torch.save(model.state_dict(), save_mode_path)

                logging.info(
                     'iteration %d : organ_dice_score : %f hd95 : %f recall : %f precision : %f jaccord : %f asd : %f'
                     % (iter_num, avg_metric[0, 0].mean(), avg_metric[0, 1].mean(), avg_metric[0, 2].mean(),
                        avg_metric[0, 3].mean(), avg_metric[0, 4].mean(), avg_metric[0, 5].mean()))
                logging.info(
                    'iteration %d : tumor_dice_score : %f hd95 : %f recall : %f precision : %f jaccord : %f asd : %f'
                    % (iter_num, avg_metric[1, 0].mean(), avg_metric[1, 1].mean(), avg_metric[1, 2].mean(),
                       avg_metric[1, 3].mean(), avg_metric[1, 4].mean(), avg_metric[1, 5].mean()))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}/{}".format(args.exp, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
