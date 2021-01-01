import os

from pytorch_lightning import LightningModule, Trainer
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


def main():
    #
    # Prepare data
    #

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = MNIST(os.getcwd(), train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64)

    #
    # Train
    #

    net = LitMNIST()
    trainer = Trainer(gpus=1)
    trainer.fit(net, train_loader)

    print('lol')

    # net = LitMNIST()
    # rand_img = torch.randn(3, 1, 28, 28)
    # out = net(rand_img)
    # print(out)


class LitMNIST(LightningModule):
    def __init__(self):
        super().__init__()

        self.linear_1 = nn.Linear(28 * 28, 128)
        self.linear_2 = nn.Linear(128, 256)
        self.linear_3 = nn.Linear(256, 10)

    def forward(self, batch):
        batch_size, channels, width, height = batch.size()

        batch_0 = batch.view(batch_size, -1)

        batch_1 = self.linear_1(batch_0)
        batch_1 = F.relu(batch_1)

        batch_2 = self.linear_2(batch_1)
        batch_2 = F.relu(batch_2)

        batch_3 = self.linear_3(batch_2)

        probs = F.softmax(batch_3, dim=1)

        return probs

    def training_step(self, batch, _):
        inputs, labels = batch

        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)

        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)


if __name__ == '__main__':
    main()
