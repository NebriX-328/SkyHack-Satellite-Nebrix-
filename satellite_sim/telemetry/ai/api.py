import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dynamic LSTM Autoencoder ---
class LSTMAutoencoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size=128, latent_size=32):
        super().__init__()
        self.encoder = torch.nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.fc1 = torch.nn.Linear(hidden_size, latent_size)
        self.fc2 = torch.nn.Linear(latent_size, hidden_size)
        self.decoder = torch.nn.LSTM(hidden_size, input_size, num_layers=2, batch_first=True)

    def forward(self, x):
        out, _ = self.encoder(x)
        latent = self.fc1(out)
        out = self.fc2(latent)
        reconstructed, _ = self.decoder(out)
        return reconstructed

# --- Dynamic model loader ---
def get_model(input_size: int, model_path: str = None):
    model = LSTMAutoencoder(input_size=input_size).to(DEVICE)
    if model_path:
        try:
            checkpoint = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(checkpoint)
        except FileNotFoundError:
            print(f"[Warning] Pre-trained model {model_path} not found. Using random weights.")
    model.eval()
    return model

# --- FastAPI app ---
app = FastAPI()

class TelemetryRequest(BaseModel):
    timestamp: str
    seq_len: int
    telemetry: Dict[str, List[float]]
    orbit: Dict
    subsystems: Dict

@app.post("/api/detect")
def api_detect(req: TelemetryRequest):
    features = list(req.telemetry.keys())
    num_features = len(features)

    # Automatically get a model that matches the number of features
    model = get_model(input_size=num_features)

    # Convert telemetry to tensor
    data = torch.tensor([list(req.telemetry.values())], dtype=torch.float32).to(DEVICE)  # [1, features, seq_len]
    data = data.permute(0, 2, 1)  # [1, seq_len, features]

    with torch.no_grad():
        reconstructed = model(data)

    # Dynamic anomaly detection: features with higher MSE
    mse_per_feature = ((data - reconstructed) ** 2).mean(dim=1).squeeze().tolist()
    anomalies = [f for i, f in enumerate(features) if mse_per_feature[i] > 0]

    core_parameters = {f: ("anomaly_detected" if f in anomalies else "normal") for f in features}

    return {
        "status": "success",
        "timestamp": req.timestamp,
        "num_anomalies": len(anomalies),
        "threshold": 0.0,
        "anomalous_features": anomalies,
        "core_parameters": core_parameters,
        "health_overview": {"status": "stable", "estimated_life_remaining": "unknown"},
        "orbit_motion": req.orbit,
        "subsystems": req.subsystems,
        "collision_debris": {"warning": "none", "details": {}},
        "graphs": req.telemetry,
        "debug": {
            "per_sequence_mse_sample": [0]*10,
            "per_feature_mse": mse_per_feature,
            "num_sequences": len(next(iter(req.telemetry.values())))
        }
    }
