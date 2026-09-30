"""
Earthquake Detection with AI (1D CNN)
-------------------------------------
1. Generates synthetic seismograms (noise vs. noise + earthquake)
2. Trains a 1D CNN classifier
3. Evaluates it
4. Scans a long continuous stream and reports detected events

Run:  python earthquake_detector.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # save plots to files (works without a display)
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

rng = np.random.default_rng(42)
tf.random.set_seed(42)

FS = 100                  # sampling rate (Hz)
WIN_SEC = 20              # window length (seconds)
WIN = FS * WIN_SEC        # samples per window


# ----------------------------------------------------------------------------
# 1. DATA GENERATION
# ----------------------------------------------------------------------------
def make_noise(n):
    """Background noise: white noise + slow microseism-like hum."""
    white = rng.normal(0, 1, n)
    white = np.convolve(white, np.ones(3) / 3, mode="same")
    t = np.arange(n) / FS
    hum = 0.5 * np.sin(2 * np.pi * rng.uniform(0.1, 0.3) * t + rng.uniform(0, 6.28))
    return white + hum


def wavelet(n, start, freq, tau, amp):
    """Decaying oscillation starting at sample `start`."""
    t = np.arange(n) / FS
    tt = np.clip(t - start / FS, 0, None)
    env = np.exp(-tt / tau) * (t >= start / FS)
    ramp = np.clip((t - start / FS) / 0.2, 0, 1)  # smooth onset
    return amp * env * ramp * np.sin(2 * np.pi * freq * tt)


def add_event(x, onset=None):
    """Add a P-wave followed by a stronger S-wave to signal x."""
    n = len(x)
    if onset is None:
        onset = int(rng.uniform(3, WIN_SEC - 8) * FS)
    p_amp = rng.uniform(2.0, 8.0)
    s_delay = int(rng.uniform(2, 5) * FS)
    p = wavelet(n, onset, rng.uniform(5, 10), rng.uniform(0.6, 1.2), p_amp)
    s = wavelet(n, onset + s_delay, rng.uniform(2, 5), rng.uniform(2, 4), p_amp * rng.uniform(1.5, 3))
    return x + p + s


def normalize(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def build_dataset(n_samples=6000):
    X, y = [], []
    for i in range(n_samples):
        w = make_noise(WIN)
        label = i % 2
        if label == 1:
            w = add_event(w)
        X.append(normalize(w))
        y.append(label)
    X = np.array(X, dtype="float32")[..., None]  # (N, WIN, 1)
    return X, np.array(y)


# ----------------------------------------------------------------------------
# 2. MODEL
# ----------------------------------------------------------------------------
def build_model():
    m = models.Sequential([
        layers.Input(shape=(WIN, 1)),
        layers.Conv1D(16, 7, activation="relu", padding="same"),
        layers.MaxPooling1D(4),
        layers.Conv1D(32, 7, activation="relu", padding="same"),
        layers.MaxPooling1D(4),
        layers.Conv1D(64, 5, activation="relu", padding="same"),
        layers.GlobalAveragePooling1D(),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m


# ----------------------------------------------------------------------------
# 3. CONTINUOUS DETECTION
# ----------------------------------------------------------------------------
def detect_stream(model, stream, step_sec=2, threshold=0.9):
    """Slide a window over the stream and return (times, probabilities, events)."""
    step = step_sec * FS
    starts = np.arange(0, len(stream) - WIN, step)
    windows = np.array([normalize(stream[s:s + WIN]) for s in starts], dtype="float32")[..., None]
    probs = model.predict(windows, verbose=0).ravel()
    times = (starts + WIN / 2) / FS

    # merge consecutive positive windows into single events
    events, in_event, begin = [], False, 0.0
    for t, p in zip(times, probs):
        if p >= threshold and not in_event:
            in_event, begin = True, t
        elif p < threshold and in_event:
            in_event = False
            events.append((begin, t))
    if in_event:
        events.append((begin, times[-1]))
    return times, probs, events


def main():
    print("Generating synthetic dataset...")
    X, y = build_dataset(6000)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    print("Training model...")
    model = build_model()
    model.fit(X_tr, y_tr, epochs=15, batch_size=64, validation_split=0.15, verbose=2)

    print("\nEvaluating on held-out test set...")
    pred = (model.predict(X_te, verbose=0).ravel() > 0.5).astype(int)
    print(classification_report(y_te, pred, target_names=["noise", "earthquake"]))
    print("Confusion matrix:\n", confusion_matrix(y_te, pred))
    model.save("earthquake_cnn.keras")
    print("Model saved to earthquake_cnn.keras")

    # Example waveforms
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    t = np.arange(WIN) / FS
    ax[0].plot(t, X_te[np.where(y_te == 0)[0][0], :, 0], lw=0.7)
    ax[0].set_title("Noise window")
    ax[1].plot(t, X_te[np.where(y_te == 1)[0][0], :, 0], lw=0.7, color="tab:red")
    ax[1].set_title("Earthquake window (P-wave then S-wave)")
    ax[1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig("examples.png", dpi=120)
    plt.close()

    # Continuous 10-minute stream with 3 hidden earthquakes
    print("\nBuilding 10-minute continuous stream with 3 hidden earthquakes...")
    total = 600 * FS
    stream = make_noise(total)
    true_onsets = [90, 260, 430]  # seconds
    for on in true_onsets:
        seg_start = on * FS
        seg = stream[seg_start:seg_start + 15 * FS].copy()
        stream[seg_start:seg_start + 15 * FS] = add_event(seg, onset=0)

    times, probs, events = detect_stream(model, stream)
    print("True event onsets (s):", true_onsets)
    print("Detected event intervals (s):")
    for b, e in events:
        print(f"  {b:6.1f} -> {e:6.1f}")

    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax[0].plot(np.arange(total) / FS, stream, lw=0.4)
    for on in true_onsets:
        ax[0].axvline(on, color="green", ls="--", label="true onset" if on == true_onsets[0] else None)
    ax[0].set_title("Continuous seismogram")
    ax[0].legend()
    ax[1].plot(times, probs, color="tab:red")
    ax[1].axhline(0.9, color="k", ls=":")
    ax[1].set_title("Model earthquake probability")
    ax[1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig("detection.png", dpi=120)
    print("Saved plots: examples.png, detection.png")


if __name__ == "__main__":
    main()
