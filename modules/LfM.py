from torch import nn


class LfM(nn.Module):
    def __init__(self, in_dim, chs):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(in_dim, 8, 4, 2, 1),
                                     nn.ReLU(),
                                     nn.Conv2d(8, 16, 4, 2, 1),
                                     nn.ReLU(),
                                     nn.Conv2d(16, 32, 4, 2, 1),
                                     nn.ReLU(),
                                     nn.Conv2d(32, 64, 4, 2, 1),
                                     nn.ReLU(),
                                     nn.Conv2d(64, 64, 4, 2, 1),
                                     nn.ReLU(),
                                     nn.Conv2d(64, 64, 4, 2, 1),
                                     nn.ReLU())

        self.decoder = nn.Sequential(nn.ConvTranspose2d(64, 64, 4, 2, 1),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(64, 64, 4, 2, 1),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(64, 32, 4, 2, 1),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(32, 16, 4, 2, 1),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(16, 8, 4, 2, 1),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(8, chs, 4, 2, 1),
                                     nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x
