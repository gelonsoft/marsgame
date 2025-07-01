import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

class StateAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed

    def encode(self, x):
        return self.encoder(x)
    
INPUT_DIM=0
data = np.load("test.npy")
tensor_data = torch.tensor(data, dtype=torch.float32)
input_dim = data.shape[1]
data=None
dataset = TensorDataset(tensor_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
device='cpu'

# Setup

latent_dim = 256  # you can tune this
autoencoder = StateAutoencoder(input_dim, latent_dim)
optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Training loop
for epoch in range(20):  # increase if needed
    epoch_loss = 0
    for batch in dataloader:
        x_batch = batch[0]
        x_recon = autoencoder(x_batch)
        loss = loss_fn(x_recon, x_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

torch.save(autoencoder.state_dict(), "state_autoencoder.pth")
#autoencoder.load_state_dict(torch.load("state_autoencoder.pth"))
#autoencoder.eval()
#with torch.no_grad():
#    compressed_data = autoencoder.encode(tensor_data).numpy()