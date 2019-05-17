import os
import copy
import json

import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import progressbar as pb

import config as cf
from dataset import MoneyFaceValue
from net import TransferLearning
import metrics as mm


def train(
    net, criterion, optimizer, dataloaders, epoch,
    device=torch.device('cuda:0'), metrics=mm.acc_t,
    best_benchmark='acc_t'
):
    # --- 整理metrics ---
    if not isinstance(metrics, (tuple, list)):
        best_benchmark = metrics.__name__
        metrics = [metrics]
    elif len(metrics) == 1:
        best_benchmark = metrics[0].__name__
    # --- 保存最好的 ---
    best_model_wts = copy.deepcopy(net.state_dict())
    best_metrics = 0.
    # --- 设备 ---
    net = net.to(device)
    criterion = criterion.to(device)
    softmax = nn.Softmax(1)
    # --- history dict ---
    indexes = ['loss'] + [m.__name__ for m in metrics]
    history = {
        phase + '_' + idx: []
        for idx in indexes for phase in ['train', 'valid']}
    for e in range(epoch):
        for phase in ['train', 'valid']:
            running_loss = 0.
            epoch_pred = []
            epoch_target = []
            if phase == 'train':
                net.train()
                prefix = 'Epoch: %d, train | ' % e
                iterator = pb.progressbar(dataloaders[phase], prefix=prefix)
            else:
                net.eval()
                iterator = dataloaders[phase]

            for img, label in iterator:
                epoch_target.append(label)
                img = img.to(device)
                label = label.to(device)
                with torch.set_grad_enabled(phase == 'train'):
                    pred = net(img)
                    loss = criterion(pred, label)
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                with torch.no_grad():
                    epoch_pred.append(softmax(pred))
                    running_loss += loss.item() * img.size(0)

            epoch_loss = running_loss / len(dataloaders[phase])
            history[phase+'_loss'].append(epoch_loss)
            epoch_pred = torch.cat(epoch_pred, dim=0)
            epoch_target = torch.cat(epoch_target, dim=0)
            epoch_metrics = {}
            for m_f in metrics:
                m_v = m_f(epoch_target, epoch_pred)
                epoch_metrics[m_f.__name__] = m_v
                history[phase+'_'+m_f.__name__].append(m_v)
            print(
                'Phase: %s, Loss: %.4f' % (phase, epoch_loss),
                *[', %s: %.4f' % (k.capitalize(), v)
                  for k, v in epoch_metrics.items()]
            )

            if (
                phase == 'valid' and
                epoch_metrics[best_benchmark] > best_metrics
            ):
                best_metrics = epoch_metrics[best_benchmark]
                best_model_wts = copy.deepcopy(net.state_dict())

    print('Best valid %s: %.4f' % (best_benchmark, best_metrics))
    net.load_state_dict(best_model_wts)

    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'name', help='本次训练的保存名称'
    )
    parser.add_argument(
        '-rs', '--random_seed', default=1234, type=int,
        help='随机种子数，默认1234'
    )
    parser.add_argument(
        '-bs', '--batch_size', default=128, type=int,
        help='批次大小，默认是128'
    )
    parser.add_argument(
        '-nj', '--n_jobs', default=6, type=int,
        help='多核并行的核数，默认是6'
    )
    parser.add_argument(
        '-lr', '--learning_rate', default=0.005, type=float,
        help='学习率，默认是0.005'
    )
    parser.add_argument(
        '-e', '--epoch', default=100, type=int,
        help="epoch数量，默认是100"
    )
    parser.add_argument(
        '-is', '--input_size', default=(300, 600), type=int, nargs=2,
        help="输入到net中的图像大小，默认是300x600(hxw)"
    )
    parser.add_argument(
        '-sd', '--save_dir', default='./results',
        help="结果保存的路径，默认是./results"
    )
    args = parser.parse_args()

    # ------- 读取并分割数据集 -------
    train_df = pd.read_csv(
        os.path.join(cf.ROOT_DIR, cf.TRAIN_LABEL_CSV))
    fs, ls = train_df.iloc[:, 0].values, train_df.iloc[:, 1].values
    trainX, validX, trainy, validy = train_test_split(
        fs, ls, test_size=0.1, shuffle=True, random_state=args.random_seed,
        stratify=ls
    )

    # ------- 构建torch dataset -------
    train_dir = os.path.join(cf.ROOT_DIR, cf.TRAIN_DIR)
    transfers = transforms.Compose([
        transforms.Resize(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    datasets = {
        'train': MoneyFaceValue(
            trainX, trainy, train_dir, label_mapper=cf.LABEL_MAPPER,
            transform=transfers
        ),
        'valid': MoneyFaceValue(
            validX, validy, train_dir, label_mapper=cf.LABEL_MAPPER,
            transform=transfers
        ),
    }
    dataloaders = {
        k: DataLoader(
            Subset(v, list(range(100))), args.batch_size,
            shuffle=(k == 'train'), num_workers=args.n_jobs
        )
        for k, v in datasets.items()
    }

    # ------- 构建网络 -------
    net = TransferLearning(models.resnet50, num_class=len(cf.LABELS))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(), lr=args.learning_rate, momentum=0.9)

    # ------- 训练网络 -------
    hist = train(
        net, criterion, optimizer, dataloaders, epoch=args.epoch,
        device=torch.device('cuda:0'), best_benchmark='acc_t',
        metrics=[mm.acc_t, mm.auc_t, mm.f1_score_t],
    )

    # ------- 保存结果 -------
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    save_name = os.path.join(args.save_dir, args.name)
    if not os.path.exists(save_name):
        os.mkdir(save_name)
    hist = pd.DataFrame(hist)
    hist.to_csv(os.path.join(save_name, 'train.csv'))
    torch.save(net.state_dict(), os.path.join(save_name, 'model.pth'))
    with open(os.path.join(save_name, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f)


if __name__ == "__main__":
    main()
