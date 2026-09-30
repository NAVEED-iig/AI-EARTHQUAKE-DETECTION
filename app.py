"""
Backend for the Earthquake AI web app.
Loads the trained model (earthquake_cnn.keras) and serves:
  GET  /             -> the frontend page
  GET  /api/health   -> is the model loaded?
  POST /api/simulate -> simulate a 10-min stream with random quakes, run detection
  POST /api/upload   -> upload a CSV/TXT waveform, run detection
  GET  /api/sample   -> download a sample CSV to test the upload
Run:  python app.py   then open http://127.0.0.1:5000
"""
import os
import numpy as np
from flask import Flask, jsonify, request, send_from_directory, Response
import tensorflow as tf

# Reuse the functions from the training script
from earthquake_detector import FS, WIN, make_noise, add_event, detect_stream, rng

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "earthquake_cnn.keras")

app = Flask(__name__, static_folder=os.path.join(BASE, "static"))

model = None
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded.")
else:
    print("WARNING: earthquake_cnn.keras not found. Run earthquake_detector.py first.")


def analyze(stream, threshold, true_onsets=None):
    """Run the model on a stream and build the JSON response."""
    times, probs, events = detect_stream(model, stream, threshold=threshold)
    out_events = []
    for b, e in events:
        mask = (times >= b) & (times <= e)
        out_events.append({"start": float(b), "end": float(e), "peak": float(probs[mask].max())})

    step = max(1, len(stream) // 6000)  # downsample for plotting
    t_axis = (np.arange(0, len(stream), step) / FS)
    return {
        "duration": len(stream) / FS,
        "t": np.round(t_axis, 2).tolist(),
        "y": np.round(stream[::step], 3).tolist(),
        "prob_t": np.round(times, 2).tolist(),
        "probs": np.round(probs, 4).tolist(),
        "events": out_events,
        "true_onsets": true_onsets or [],
        "threshold": threshold,
    }


def make_stream(duration_s, n_events):
    """Noise stream with n_events random earthquakes, at least 60 s apart."""
    total = duration_s * FS
    stream = make_noise(total)
    onsets = []
    tries = 0
    while len(onsets) < n_events and tries < 500:
        tries += 1
        c = int(rng.integers(20, duration_s - 30))
        if all(abs(c - o) >= 60 for o in onsets):
            onsets.append(c)
    onsets.sort()
    for on in onsets:
        s = on * FS
        stream[s:s + 15 * FS] = add_event(stream[s:s + 15 * FS].copy(), onset=0)
    return stream, onsets


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/health")
def health():
    return jsonify({"model_loaded": model is not None, "sampling_rate": FS, "window_seconds": WIN // FS})


@app.route("/api/simulate", methods=["POST"])
def simulate():
    if model is None:
        return jsonify({"error": "Model not found. Run earthquake_detector.py first."}), 500
    body = request.get_json(silent=True) or {}
    n_events = int(np.clip(body.get("n_events", 3), 0, 5))
    threshold = float(np.clip(body.get("threshold", 0.9), 0.05, 0.99))
    stream, onsets = make_stream(600, n_events)
    return jsonify(analyze(stream, threshold, onsets))


@app.route("/api/upload", methods=["POST"])
def upload():
    if model is None:
        return jsonify({"error": "Model not found. Run earthquake_detector.py first."}), 500
    f = request.files.get("file")
    if f is None:
        return jsonify({"error": "No file uploaded."}), 400
    threshold = float(np.clip(float(request.form.get("threshold", 0.9)), 0.05, 0.99))
    fs_in = float(request.form.get("fs", FS))
    if fs_in <= 0:
        return jsonify({"error": "Sampling rate must be positive."}), 400

    # Read the LAST column of every line (so "time,amplitude" or a single column both work)
    values = []
    for line in f.read().decode("utf-8", errors="ignore").splitlines():
        parts = line.replace(";", ",").replace("\t", ",").split(",")
        try:
            values.append(float(parts[-1]))
        except ValueError:
            continue  # skip headers / blank lines
    x = np.array(values, dtype="float64")

    if fs_in != FS:  # resample to the 100 Hz the model was trained on
        new_n = int(len(x) * FS / fs_in)
        x = np.interp(np.linspace(0, len(x) - 1, new_n), np.arange(len(x)), x)

    if len(x) < WIN + FS:
        return jsonify({"error": f"Need at least {(WIN + FS) / FS:.0f} seconds of data "
                                 f"(got {len(x) / FS:.1f} s after resampling)."}), 400
    x = x - x.mean()
    return jsonify(analyze(x, threshold))


@app.route("/api/sample")
def sample():
    stream, _ = make_stream(120, 1)
    csv = "amplitude\n" + "\n".join(f"{v:.4f}" for v in stream)
    return Response(csv, mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=sample_quake.csv"})


if __name__ == "__main__":
    # debug=False so TensorFlow only loads once
    app.run(host="127.0.0.1", port=5000, debug=False)
