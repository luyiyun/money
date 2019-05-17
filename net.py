import torch.nn as nn


class TransferLearning(nn.Module):

    def __init__(self, backbone, num_class, pretrained=True):
        super(TransferLearning, self).__init__()
        self.backbone = backbone(pretrained)
        self.backbone_layers = nn.Sequential(
            *list(self.backbone.children())[:-1])
        self.fc = nn.Linear(2048, num_class)

    def forward(self, x):
        x = self.backbone_layers(x)
        return self.fc(x.squeeze())


def test():
    import torch
    from torchvision.models import resnet50

    x = torch.rand(2, 3, 224, 224)
    net = TransferLearning(resnet50, 9)
    y = net(x)
    print(y.shape)


if __name__ == "__main__":
    test()
