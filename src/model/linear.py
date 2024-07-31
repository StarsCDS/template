# References:
## build model: https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
## save/load: https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import typer
from rich.progress import track
from src.data.fmnist import FMNIST

app = typer.Typer()


class Linear(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


def train_epoch(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (features, target) in enumerate(dataloader):
        features, target = features.to(DEVICE), target.to(DEVICE)

        pred = model(features)
        loss = loss_fn(pred, target)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(features)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test_epoch(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    n_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for features, target in dataloader:
            features, target = features.to(DEVICE), target.to(DEVICE)

            pred = model(features)
            test_loss += loss_fn(pred, target).item()
            correct += (pred.argmax(1) == target).type(torch.float).sum().item()
            test_loss /= num_batches
            correct /= size

            print(
                f"Test Error: \nAccuracy: {(100 * correct):>0.1f}%",
                "Avg loss: {test_loss:>8f} \n",
            )


# ============================= Main ==================================

SAVED_MODEL = "linear_model.pth"
DEVICE = "cpu"
BATCH_SIZE = 256
EPOCHS = 5

data_train, data_test = FMNIST(), FMNIST(train=False)
loader_train = DataLoader(data_train, batch_size=BATCH_SIZE)
loader_test = DataLoader(data_test, batch_size=BATCH_SIZE)
model = Linear(data_train.sizes["input"], data_train.sizes["output"]).to(DEVICE)

optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
loss_fn = nn.functional.mse_loss


@app.command()
def train():
    for _ in track(range(epochs)):
        train_epoch(loader_train, model, loss_fn, optimizer)
        test_epoch(test_dataloader, model, loss_fn)

    torch.save(model.state_dict(), SAVED_MODEL)


@app.command()
def evalualte():
    model.load_state_dict(torch.load(SAVED_MODEL))
    model.eval()


if __name__ == "__main__":
    app()
