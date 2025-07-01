import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, IterableDataset
import torch.optim as optim

class StreamingCSVDataset(IterableDataset):
    def __init__(self, filename, scaler=None, skip_header=True, chunk_size=1024):
        self.filename = filename
        self.scaler = scaler
        self.skip_header = skip_header
        self.chunk_size = chunk_size

    def __iter__(self):
        chunk_iter = pd.read_csv(self.filename, chunksize=self.chunk_size, iterator=True)
        for chunk in chunk_iter:
            data = chunk.to_numpy().astype(np.float32)
            if self.scaler:
                data = self.scaler.transform(data)
            for row in data:
                yield torch.tensor(row)

    def get_sample_for_scaler(self, sample_rows=1000):
        df = pd.read_csv(self.filename, nrows=sample_rows)
        return df.to_numpy().astype(np.float32)
    
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
csv_path = "big_file.csv"
temp_dataset = StreamingCSVDataset(csv_path)
sample_data = temp_dataset.get_sample_for_scaler()
scaler = MinMaxScaler()
scaler.fit(sample_data)
streaming_dataset = StreamingCSVDataset(csv_path, scaler=scaler)
dataloader = DataLoader(streaming_dataset, batch_size=64)
print("Data loaded")

device='cpu'

# Setup

latent_dim = 256  # you can tune this
autoencoder = StateAutoencoder(input_dim=sample_data.shape[1], latent_dim=16)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

for epoch in range(10):
    total_loss = 0
    for batch in dataloader:
        x = batch
        x_recon = autoencoder(x)
        loss = loss_fn(x_recon, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

torch.save(autoencoder.state_dict(), "state_autoencoder.pth")
#autoencoder.load_state_dict(torch.load("state_autoencoder.pth"))
#autoencoder.eval()
#with torch.no_grad():
#    compressed_data = autoencoder.encode(tensor_data).numpy()