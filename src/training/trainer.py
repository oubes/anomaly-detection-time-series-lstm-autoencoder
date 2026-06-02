import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def train(model, x_train, config, device):

    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=config["lr"])

    loss_fn = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(torch.tensor(x_train)),
        batch_size=config["batch_size"]
    )

    train_losses = []

    for ep in range(config["epochs"]):

        model.train()
        total = 0

        for (b,) in loader:
            b = b.to(device)

            loss = loss_fn(model(b), b)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        avg = total / len(loader)
        train_losses.append(avg)

        print(f"Epoch {ep+1} | loss={avg:.5f}")

    return model, train_losses