import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# -------------------------------
# Config (match train.py)
# -------------------------------
SEQ_LEN = 100
BATCH_SIZE = 64
MODEL_FOLDER = 'models'
DATA_FOLDER_25 = 'data/test_25'
DATA_FOLDER_55 = 'data/test_55'
OUTPUT_FOLDER = 'results'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

HIDDEN_DIM = 128
LATENT_DIM = 32
NUM_LAYERS = 2
DROPOUT = 0.2

# -------------------------------
# Dataset
# -------------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, folder_path, seq_len=SEQ_LEN):
        self.seq_len = seq_len
        self.data = []

        for f in os.listdir(folder_path):
            if f.endswith('.npy'):
                arr = np.load(os.path.join(folder_path, f))
                scaler = MinMaxScaler()
                arr = scaler.fit_transform(arr)
                sequences = [arr[i:i+seq_len] for i in range(len(arr)-seq_len+1)]
                self.data.extend(sequences)

        self.data = np.array(self.data, dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)

# -------------------------------
# LSTM Autoencoder
# -------------------------------
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers=num_layers, batch_first=True, dropout=dropout)

    def forward(self, x):
        out, _ = self.encoder(x)
        latent = self.fc1(out[:, -1, :])
        out = self.fc2(latent).unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.decoder(out)
        return out

# -------------------------------
# Test function
# -------------------------------
def test_model(model_path, input_dim, data_folder, seq_len=SEQ_LEN):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMAutoencoder(input_dim).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    dataset = TimeSeriesDataset(data_folder, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_losses = []
    criterion = nn.MSELoss(reduction='none')
    reconstructed = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            output = model(batch)
            loss = criterion(output, batch).mean(dim=(1, 2))
            all_losses.extend(loss.cpu().numpy())
            reconstructed.extend(output.cpu().numpy())

    all_losses = np.array(all_losses)
    reconstructed = np.array(reconstructed)

    # Save .npy files
    np.save(os.path.join(OUTPUT_FOLDER, f'reconstruction_{input_dim}f.npy'), reconstructed)
    np.save(os.path.join(OUTPUT_FOLDER, f'losses_{input_dim}f.npy'), all_losses)

    # Dynamic threshold
    threshold = all_losses.mean() + 3 * all_losses.std()
    anomalies = [i for i, l in enumerate(all_losses) if l > threshold]
    np.save(os.path.join(OUTPUT_FOLDER, f'anomalies_{input_dim}f.npy'), anomalies)

    return anomalies, all_losses, threshold

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    print("Testing 25-feature model...")
    anomalies_25, losses_25, threshold_25 = test_model(
        os.path.join(MODEL_FOLDER, 'lstm_autoencoder_25f.pt'), 25, DATA_FOLDER_25
    )
    print(f"Detected {len(anomalies_25)} anomalies (threshold={threshold_25:.6f})")

    print("Testing 55-feature model...")
    anomalies_55, losses_55, threshold_55 = test_model(
        os.path.join(MODEL_FOLDER, 'lstm_autoencoder_55f.pt'), 55, DATA_FOLDER_55
    )
    print(f"Detected {len(anomalies_55)} anomalies (threshold={threshold_55:.6f})")
    

def run_inference(input_sequence):
    """
    Load the trained LSTM Autoencoder model and perform prediction.
    """
    model = torch.load("telemetry/ai/models/lstm_autoencoder_25f.pt", map_location="cpu")
    model.eval()

    input_tensor = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        reconstructed = model(input_tensor)
        loss = torch.mean((input_tensor - reconstructed) ** 2).item()

    # Define simple anomaly threshold
    is_anomaly = loss > 0.05
    return {"loss": round(loss, 6), "anomaly_detected": is_anomaly}
