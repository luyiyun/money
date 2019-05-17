import os

import pandas as pd
from torch.utils import data
from PIL import Image


def pil_loader(f):
    return Image.open(f)


class MoneyFaceValue(data.Dataset):
    def __init__(
        self, files, labels, files_dir, transform=None, loader=pil_loader,
        label_mapper=None
    ):
        super(MoneyFaceValue, self).__init__()
        self.files = files
        self.labels = labels
        self.files_dir = files_dir
        self.loader = loader
        self.transform = transform
        self.label_mapper = label_mapper

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f, label = self.files[idx], self.labels[idx]
        img = self.loader(os.path.join(self.files_dir, f))
        if self.transform is not None:
            img = self.transform(img)
        if self.label_mapper is not None:
            label = self.label_mapper[label]
        return img, label


def test():
    import numpy as np
    import matplotlib.pyplot as plt
    import config as cf

    train_df = pd.read_csv(
        os.path.join(cf.ROOT_DIR, cf.TRAIN_LABEL_CSV))
    fs, ls = train_df.iloc[:, 0].values, train_df.iloc[:, 1].values
    datasets = MoneyFaceValue(
        fs, ls, os.path.join(cf.ROOT_DIR, cf.TRAIN_DIR),
        label_mapper=cf.LABEL_MAPPER
    )
    for img, label in datasets:
        img = np.array(img)
        plt.imshow(img)
        print(label)
        plt.show()


if __name__ == "__main__":
    test()
