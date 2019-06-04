import os

import pandas as pd
import argparse
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
from torchvision import models
import progressbar as pb

from dataset import pil_loader
from net import TransferLearning
import config as cf


def predict(net, dataloader, device):
    net = net.to(device)
    softmax = nn.Softmax(1)
    preds = []
    fns = []
    for img, fn in pb.progressbar(dataloader):
        img = img.to(device)
        pred = net(img)
        pred = softmax(pred)
        _, pred = pred.max(1)
        preds.append(pred)
        fns.extend(list(fn))
    preds = torch.cat(preds, dim=0)
    return preds, fns


class PredData(data.Dataset):
    def __init__(
        self, imgdir, imgtype='jpg', transform=None, loader=pil_loader
    ):
        self.transform = transform
        self.loader = pil_loader
        self.filenames = [f for f in os.listdir(imgdir) if f.endswith(imgtype)]
        self.files = [os.path.join(imgdir, f) for f in self.filenames]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f, fn = self.files[idx], self.filenames[idx]
        img = self.loader(f)
        if self.transform is not None:
            img = self.transform(img)
        return img, fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'imgs', default='./public_test_data', nargs='?',
        help='要预测的数据集所在的文件夹，默认是./public_test_data'
    )
    parser.add_argument(
        '-md', '--model_dir', default='./results/res1',
        help="数据保存的文件夹，默认是./results/res1"
    )
    parser.add_argument(
        '-bs', '--batch_size', default=16, type=int,
        help='批次大小，默认是16'
    )
    parser.add_argument(
        '-nj', '--n_jobs', default=6, type=int,
        help='多核并行的核数，默认是6'
    )
    parser.add_argument(
        '-is', '--input_size', default=(300, 600), type=int, nargs=2,
        help="输入到net中的图像大小，默认是300x600(hxw)"
    )
    parser.add_argument(
        '-sf', '--save_file', default='./results/res1/test.csv',
        help="保存的结果的csv文件，默认是./results/res1/test.csv"
    )
    args = parser.parse_args()

    # ------- 构建torch dataset -------
    transfers = transforms.Compose([
        transforms.Resize(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = PredData(args.imgs, transform=transfers)
    dataloader = data.DataLoader(
        dataset, args.batch_size, num_workers=args.n_jobs)

    # ------- 载入训练好的模型 -------
    state_dict = torch.load(os.path.join(args.model_dir, 'model.pth'))
    net = TransferLearning(models.resnet50, num_class=len(cf.LABELS))
    net.load_state_dict(state_dict)

    # ------- 预测结果，并保存 -------
    preds, fns = predict(net, dataloader, device=torch.device('cuda:0'))
    inv_label_mapper = {
        v: str(k) if k < 1 else str(int(k))
        for k, v in cf.LABEL_MAPPER.items()
    }
    preds = [inv_label_mapper[p] for p in preds.cpu().numpy()]
    df = pd.DataFrame({'name': fns, 'label': preds})
    df.to_csv(args.save_file, index=False)


if __name__ == "__main__":
    main()
