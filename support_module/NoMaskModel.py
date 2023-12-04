from torch import nn
import logging


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
        self.dropout = nn.Dropout(0.5)

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()
        self.input_liner = nn.Linear(32 * 64 * 64, 2048)
        self.backend_liner1 = nn.Linear(2048, 256)
        self.output_liner = nn.Linear(256, 3)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        logging.info("Starting conv1...")
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool(self.relu(x))

        logging.info("Starting conv2...")
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool(self.relu(x))

        # Выравнивание тензора перед подачей на полносвязный слой
        x = x.view(-1, 32 * 64 * 64)

        logging.info("Starting liner1...")
        x = self.relu(self.input_liner(x))
        x = self.dropout(x)

        logging.info("Starting liner2...")
        x = self.relu(self.backend_liner1(x))
        x = self.dropout(x)

        logging.info("Starting liner3...")
        x = self.output_liner(x)

        # Применение Softmax для получения вероятностного распределения
        x = self.softmax(x)
        return x


if __name__ == '__main__':
    from torchsummary import summary

    model = NoMaskModel()
    summary(model, (1, 256, 256))
