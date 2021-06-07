import torch
from torch import nn


class SoftGatedHG(nn.Module):
    def __init__(self, in_channels, out_channels=1, time_channel=False, pred_radius=False, device="cuda"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_channel = time_channel
        self.pred_radius = pred_radius
        self.device = device

        self.encoder = [nn.Sequential(nn.Conv2d(in_channels, 8, 4, 2, 1),
                                      nn.ReLU()),
                        nn.Sequential(nn.Conv2d(8, 16, 4, 2, 1),
                                      nn.ReLU()),
                        nn.Sequential(nn.Conv2d(16, 32, 4, 2, 1),
                                      nn.ReLU()),
                        nn.Sequential(nn.Conv2d(32, 64, 4, 2, 1),
                                      nn.ReLU()),
                        nn.Sequential(nn.Conv2d(64, 64, 4, 2, 1),
                                      nn.ReLU()),
                        nn.Sequential(nn.Conv2d(64, 64, 4, 2, 1),
                                      nn.ReLU())]

        self.flatten = nn.Flatten()

        self.decoder = [nn.Sequential(nn.ConvTranspose2d(64, 64, 4, 2, 1),
                                      nn.ReLU()),
                        nn.Sequential(nn.ConvTranspose2d(64, 64, 4, 2, 1),
                                      nn.ReLU()),
                        nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1),
                                      nn.ReLU()),
                        nn.Sequential(nn.ConvTranspose2d(32, 16, 4, 2, 1),
                                      nn.ReLU()),
                        nn.Sequential(nn.ConvTranspose2d(16, 8, 4, 2, 1),
                                      nn.ReLU()),
                        nn.Sequential(nn.ConvTranspose2d(8, out_channels, 4, 2, 1),
                                      nn.ReLU())]

        assert len(self.encoder) == len(self.decoder)

        self.skip_layers = [nn.Conv2d(8, 8, 3, padding=1),
                            nn.Conv2d(16, 16, 3, padding=1),
                            nn.Conv2d(32, 32, 3, padding=1),
                            nn.Conv2d(64, 64, 3, padding=1),
                            nn.Conv2d(64, 64, 3, padding=1)]

        if self.pred_radius:
            self.radius_head = nn.Sequential(nn.Linear(64, 32),
                                             nn.ReLU(),
                                             nn.Linear(32, 1),
                                             nn.ReLU())

        self.to_device()

    def to_device(self):
        for layer in self.encoder:
            layer.to(self.device)

        self.flatten.to(self.device)

        for layer in self.decoder:
            layer.to(self.device)

        for layer in self.skip_layers:
            layer.to(self.device)

        if self.pred_radius:
            self.radius_head.to(self.device)

    def forward(self, x, time_channel=None):

        if self.time_channel:
            x = torch.cat([x, time_channel], dim=1)

        encodings = []
        for layer in self.encoder:
            x = layer(x)
            encodings.append(x)

        if self.pred_radius:
            r = self.radius_head(self.flatten(x))

        num_encodings = len(encodings)

        for i, layer in enumerate(self.decoder):
            x = self.skip_layers[num_encodings-i-2](encodings[num_encodings-i-2]) + layer(x)

        if self.pred_radius:
            return x, r
        return x
