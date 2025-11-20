import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# -------------------------------
# Config
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# Test Function
# -------------------------------
def test_model(model_path, input_dim, data_folder):
    model = LSTMAutoencoder(input_dim=input_dim).to(device)

    # Load weights safely
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    dataset = TimeSeriesDataset(data_folder)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_losses = []
    criterion = nn.MSELoss(reduction='none')

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            output = model(batch)
            loss = criterion(output, batch).mean(dim=(1, 2))
            all_losses.extend(loss.cpu().numpy())

    threshold = np.mean(all_losses) + 3 * np.std(all_losses)
    preds = np.array([1 if l > threshold else 0 for l in all_losses])

    return preds, np.array(all_losses), threshold

# -------------------------------
# Ground truth from CSV
# -------------------------------
def load_ground_truth(csv_file, data_folder):
    df = pd.read_csv(csv_file)
    total_len = sum([len(np.load(os.path.join(data_folder, f))) for f in os.listdir(data_folder) if f.endswith('.npy')])
    gt = np.zeros(total_len, dtype=int)

    for _, row in df.iterrows():
        seqs = eval(row['anomaly_sequences'])
        for s in seqs:
            start, end = s
            gt[start:end+1] = 1
    return gt

# -------------------------------
# Evaluation
# -------------------------------
def evaluate_model(model_path, test_folder, csv_file):
    input_dim = 25 if "25" in model_path else 55
    preds, losses, threshold = test_model(model_path, input_dim, test_folder)
    gt = load_ground_truth(csv_file, test_folder)

    # Ensure same length
    min_len = min(len(gt), len(preds))
    gt, preds, losses = gt[:min_len], preds[:min_len], losses[:min_len]

    # Flatten
    gt = np.ravel(gt).astype(int)
    preds = np.ravel(preds).astype(int)

    # Compute metrics
    precision = precision_score(gt, preds, zero_division=0)
    recall = recall_score(gt, preds, zero_division=0)
    f1 = f1_score(gt, preds, zero_division=0)
    auc = roc_auc_score(gt, losses) if len(np.unique(gt)) > 1 else 0

    print(f"Model: {input_dim}f")
    print(f"Detected {np.sum(preds)} anomalies (threshold={threshold:.6f})")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, AUC: {auc:.4f}")
    print(f"Mean Loss: {losses.mean():.6f}, Max Loss: {losses.max():.6f}\n")

    return preds, gt, losses, threshold, precision, recall, f1, auc

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    csv_file = 'data/labeled_anomalies.csv'

    print("Evaluating 25-feature model...")
    evaluate_model(os.path.join(MODEL_FOLDER, 'lstm_autoencoder_25f.pt'),
                   DATA_FOLDER_25, csv_file)

    print("Evaluating 55-feature model...")
    evaluate_model(os.path.join(MODEL_FOLDER, 'lstm_autoencoder_55f.pt'),
                   DATA_FOLDER_55, csv_file)
