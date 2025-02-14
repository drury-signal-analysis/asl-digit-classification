import torch
from torch.utils.data import DataLoader, Subset
from torch.backends import mps
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import random_split
from typing import Any

PARTITION = 0.8
MOMENTUM = 0.9
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 32


def subset_data(examples: torchvision.datasets.ImageFolder, partition: float) -> tuple[Subset[Any], Subset[Any]]:
    example_len = len(examples)
    train_size = int(example_len * partition)

    train_data, test_data = random_split(examples, (train_size, example_len - train_size))

    print(f'training on: {len(train_data)}, testing on: {len(test_data)}')

    return train_data, test_data


transform = transforms.Compose([transforms.ToTensor()])

full_dataset = torchvision.datasets.ImageFolder(root='images', transform=transform)
training_data, testing_data = subset_data(full_dataset, PARTITION)


training_data_loader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
testing_data_loader = DataLoader(testing_data, batch_size=BATCH_SIZE, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'mps' if mps.is_available() else 'cpu'


class ConvolutionalModel(nn.Module):
    def __init__(self):
        super(ConvolutionalModel, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=50, out_channels=20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Flatten(),
            nn.Linear(50000, 10),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, forward_xss):
        return self.network(forward_xss)


model = ConvolutionalModel().to(device)
criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)


def train(data_loader: DataLoader, model: ConvolutionalModel, loss_fn: nn.NLLLoss, optimizer: torch.optim.SGD) -> None:
    model.train()

    size = len(data_loader.dataset)

    for batch, (images, labels) in enumerate(data_loader):
        images = images.to(device)
        labels = labels.to(device)

        predictions = model(images)
        loss = loss_fn(predictions, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f'loss: {loss.item():>7f}  [{batch * BATCH_SIZE:>5d}/{size}]')


def test(data_loader: DataLoader, model: ConvolutionalModel, loss_fn: nn.NLLLoss) -> None:
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images)

            test_loss += loss_fn(predictions, labels).item()

            for prediction, label in zip(predictions, labels):
                if torch.argmax(prediction).item() == label.item():
                    correct += 1

    test_loss /= len(data_loader)
    correct /= len(data_loader.dataset)

    print(f'Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n')


print(f'Training on {device}\n')

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}\n-------------------------------')
    train(training_data_loader, model, criterion, optimizer)
    test(testing_data_loader, model, criterion)

print('Done!')

torch.jit.script(model).save('models/01.pt')
