import numpy as np
import torch
from torch import nn
import torch.functional as T
import matplotlib.pyplot as plt


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            # input is 7806 x 1
            nn.Linear(in_features=7806, out_features=4086),
            # nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(in_features=4086, out_features=2048),
            # nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=2048),
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=2048),
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=2),
        )

    def forward(self, input):
        return self.model(input)


class ClusterDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

        assert self.X.shape[0] == self.y.shape[0]

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]

        return X, y

    def __len__(self):
        return len(self.X)


class RegionDiscriminator:
    def __init__(self, cluster_name, data, batch_size=32, device='cuda'):
        super(RegionDiscriminator, self).__init__()
        self.cluster_name = cluster_name
        self.data = data[cluster_name]
        self.model = Model().to(device)
        self.device = device

        self.X = np.concatenate([self.data["pos"], self.data["neg"]], axis=0)
        self.y = np.concatenate([np.ones((self.data["pos"].shape[0],)), np.zeros((self.data["neg"].shape[0],))],
                                axis=0)

        self.dataset = ClusterDataset(self.X, self.y)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

    def train(self, nepochs=100):

        loss_fn = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        epoch_losses = []

        for epoch in range(1, nepochs + 1):
            epoch_loss = 0
            for i, batch in enumerate(self.dataloader):
                X, y = batch[0].float().to(self.device), batch[1].long().to(self.device)
                logits = self.model(X)

                loss = loss_fn(logits, y)

                opt.zero_grad()
                loss.backward()
                opt.step()

                print(f"Epoch: {epoch} | Iteration: {i} | loss: {loss.item()}")
                epoch_loss += loss.item()

            epoch_loss /= len(self.dataloader)

            epoch_losses.append(epoch_loss)

        plt.plot(list(range(1, nepochs+1)), epoch_losses)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()
