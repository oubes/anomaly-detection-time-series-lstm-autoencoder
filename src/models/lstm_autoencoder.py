import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):

    def __init__(self, input_dim=1, hidden_dim=64, latent_dim=16, num_layers=1):
        super().__init__()

        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)

        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):

        _, (h, _) = self.encoder(x)

        z = self.to_latent(h[-1])

        h_dec = self.from_latent(z)

        seq_len = x.shape[1]
        dec_in = h_dec.unsqueeze(1).repeat(1, seq_len, 1)

        h0 = h_dec.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)

        out, _ = self.decoder(dec_in, (h0, c0))

        return self.output_layer(out)