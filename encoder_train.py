print("Started 0 ",flush=True)
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, IterableDataset
import torch.optim as optim

class StreamingCSVDataset(IterableDataset):
    def __init__(self, filename, scaler=None, skip_header=True, chunk_size=100):
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

    def get_sample_for_scaler(self, sample_rows=100):
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
    
print("Started",flush=True)
INPUT_DIM=0
csv_path = "encoder_train.csv"
temp_dataset = StreamingCSVDataset(csv_path)
print("StreamingCSVDataset",flush=True)
sample_data = temp_dataset.get_sample_for_scaler()
print("sample_data",flush=True)
temp_dataset=None

scaler = MinMaxScaler()
scaler.fit(sample_data)
input_dim=sample_data.shape[1]
sample_data=None
streaming_dataset = StreamingCSVDataset(csv_path, scaler=scaler)
dataloader = DataLoader(streaming_dataset, batch_size=64)
print("Data loaded",flush=True)

device='cpu'

# Setup

latent_dim = 512  # you can tune this
autoencoder = StateAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

print("Training started",flush=True)
for epoch in range(100):
    total_loss = 0
    for batch in dataloader:
        print(".",end=None,flush=True)
        x = batch
        x_recon = autoencoder(x)
        loss = loss_fn(x_recon, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"\nEpoch {epoch+1}, Loss: {total_loss:.4f}",flush=True)

torch.save(autoencoder.state_dict(), "state_autoencoder.pth")
print("Done",flush=True)
#autoencoder.load_state_dict(torch.load("state_autoencoder.pth"))
#autoencoder.eval()
#with torch.no_grad():
#    compressed_data = autoencoder.encode(tensor_data).numpy()
#git pull ; docker run --rm --name tms-bot-train -v $(pwd):/data -e SERVER_BASE_URL="${SERVER_BASE_URL}" -e START_LR="${START_LR}" -e CONTINUE_TRAIN="${CONTINUE_TRAIN}" -e MODEL_PATH="${MODEL_PATH}" -e RUN_NAME="${RUN_NAME}"  terraforming-mars-bot:latest  python3 encoder_train.py