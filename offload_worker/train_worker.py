import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

import fl_pb2


# ---------------------------
# Model Architecture
# ---------------------------

class SimpleNN(nn.Module):

    def __init__(self):
        super(SimpleNN, self).__init__()

        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def create_model():
    return SimpleNN()


# ---------------------------
# Load Dataset
# ---------------------------

def get_dataloader():

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )

    return loader


# ---------------------------
# Training
# ---------------------------

def train_model(model, epochs=1):

    loader = get_dataloader()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()

    for epoch in range(epochs):

        for images, labels in loader:

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

    return model


# ---------------------------
# Weight Helpers
# ---------------------------

def get_model_weights(model):

    weights = {}

    for name, param in model.state_dict().items():

        weights[name] = fl_pb2.Weights(
            values=param.cpu().numpy().flatten().tolist()
        )

    return weights


def set_model_weights(model, weights):

    state_dict = model.state_dict()

    for name in state_dict:

        if name in weights:

            values = weights[name].values

            tensor = torch.tensor(values).view(state_dict[name].shape)

            state_dict[name] = tensor

    model.load_state_dict(state_dict)