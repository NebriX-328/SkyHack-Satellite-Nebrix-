import os
import numpy as np
from typing import List, Optional

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse(os.path.join("static", "index1.html"))

@app.get("/default_animation")
def default_animation():
    # return JSON telemetry data
    return {"telemetry": np.random.rand(50,3).tolist()}
