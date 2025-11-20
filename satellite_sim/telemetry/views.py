from django.shortcuts import render

def index(request):
    return render(request, 'telemetry/index.html')

def index1(request):
    return render(request, 'static/index1.html')

def dashboard(request):
    return render(request, 'telemetry/dashboard.html')

def simulator(request):
    return render(request, 'telemetry/simulator.html')

import torch
import numpy as np
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .ai.evaluate import evaluate_model
from .ai.test import run_inference

@csrf_exempt
def predict_anomaly(request):
    """
    Accepts telemetry JSON or CSV upload and returns model inference.
    """
    if request.method == "POST":
        try:
            # Parse JSON telemetry data
            data = json.loads(request.body)
            input_sequence = np.array(data["telemetry"]).astype(np.float32)

            # Call your test.py or evaluate.py logic
            result = run_inference(input_sequence)

            return JsonResponse({"status": "success", "result": result})
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)})
    else:
        return JsonResponse({"message": "Send POST request with telemetry JSON."})

import numpy as np
import datetime
import os

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@csrf_exempt
def process_npy(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    npy_file = request.FILES.get("npyFile")
    if not npy_file:
        return JsonResponse({"error": "No file uploaded"}, status=400)

    # Save uploaded .npy file
    filepath = os.path.join(UPLOAD_DIR, npy_file.name)
    with open(filepath, "wb+") as dest:
        for chunk in npy_file.chunks():
            dest.write(chunk)

    # Load numpy data
    try:
        data = np.load(filepath)
    except Exception as e:
        return JsonResponse({"error": f"Failed to load .npy file: {str(e)}"}, status=500)

    # Example anomaly detection logic
    threshold = 0.012345
    anomalies = np.where(data > threshold)[0]
    result = {
        "status": "success",
        "timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "num_anomalies": len(anomalies),
        "threshold": threshold,
        "anomalous_features": [f"feature{i}" for i in anomalies[:5].tolist()]  # top 5 anomalies
    }

    return JsonResponse(result)
