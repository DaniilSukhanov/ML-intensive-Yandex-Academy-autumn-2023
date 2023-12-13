from torch import nn
import logging
import datetime


class NoMaskModel(nn.Module):
    dropout: nn.Dropout
    input_liner: nn.Linear
    softmax: nn.Softmax
    output_liner: nn.Linear
    relu: nn.ReLU
    pool: nn.MaxPool2d
    conv1: nn.Conv2d
    conv2: nn.Conv2d
    b1: nn.BatchNorm2d
    b2: nn.BatchNorm2d
    backend_liner1: nn.Linear

    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.2)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.conv1.out_channels, out_channels=12, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(self.conv1.out_channels)
        self.bn2 = nn.BatchNorm2d(self.conv2.out_channels)

        self.relu = nn.ReLU()
        self.input_liner = nn.Linear(self.conv2.out_channels * 64 * 64, 128)
        self.output_liner = nn.Linear(self.input_liner.out_features, 3)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool(self.relu(x))

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool(self.relu(x))

        # Выравнивание тензора перед подачей на полносвязный слой
        x = x.view(-1, self.conv2.out_channels * 64 * 64)

        x = self.relu(self.input_liner(x))
        x = self.dropout(x)

        x = self.output_liner(x)

        # Применение Softmax для получения вероятностного распределения
        x = self.softmax(x)

        return x


if __name__ == '__main__':
    from torchsummary import summary

    model = NoMaskModel()
    summary(model, (1, 256, 256))
