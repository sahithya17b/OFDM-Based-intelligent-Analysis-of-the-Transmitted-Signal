from flask import Flask, request, redirect, url_for, session, jsonify, render_template_string
import numpy as np
import math
import random
import time
import threading
import webbrowser

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


# ============================================================
# NEXUS-WAVE AI
# Intelligent 3D OFDM Communication & Adaptive Equalization
# ============================================================

app = Flask(__name__)
app.secret_key = "NEXUS_WAVE_AI_2026_SECRET_KEY"

USERNAME = "admin"
PASSWORD = "ofdm2026"

N_SUBCARRIERS = 64
CP_LENGTH = 16
PATH_GAINS = np.array([0.9, 0.4, 0.2])
TRAIN_SNR = [0, 5, 10, 15, 20]
TEST_SNR = [-5, 0, 5, 10, 15, 20, 25]

AI_MODEL = None
INPUT_SCALER = None
OUTPUT_SCALER = None


# ============================================================
# COMMUNICATION FUNCTIONS
# ============================================================

def qpsk_mod(bits):
    bits = np.asarray(bits).reshape(-1, 2)

    mapping = {
        (0, 0): 1 + 1j,
        (0, 1): 1 - 1j,
        (1, 1): -1 - 1j,
        (1, 0): -1 + 1j
    }

    symbols = np.array(
        [mapping[tuple(b)] for b in bits],
        dtype=complex
    )

    return symbols / np.sqrt(2)


def qpsk_demod(symbols):
    bits = []

    for s in symbols:
        real = np.real(s)
        imag = np.imag(s)

        if real >= 0 and imag >= 0:
            bits.extend([0, 0])
        elif real >= 0 and imag < 0:
            bits.extend([0, 1])
        elif real < 0 and imag < 0:
            bits.extend([1, 1])
        else:
            bits.extend([1, 0])

    return np.array(bits, dtype=int)


def ofdm_transmit(symbols):
    time_signal = np.fft.ifft(symbols, n=N_SUBCARRIERS)

    cp = time_signal[-CP_LENGTH:]

    return np.concatenate([cp, time_signal])


def ofdm_receive(signal):
    useful = signal[CP_LENGTH:CP_LENGTH + N_SUBCARRIERS]

    return np.fft.fft(useful, n=N_SUBCARRIERS)


def multipath_channel():
    taps = np.zeros(3, dtype=complex)

    for i, gain in enumerate(PATH_GAINS):
        phase = np.random.uniform(0, 2 * np.pi)
        taps[i] = gain * np.exp(1j * phase)

    return taps


def apply_channel(signal, taps):
    return np.convolve(signal, taps, mode="full")[:len(signal)]


def add_awgn(signal, snr_db):
    power = np.mean(np.abs(signal) ** 2)

    snr_linear = 10 ** (snr_db / 10)

    noise_power = power / snr_linear

    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(len(signal))
        + 1j * np.random.randn(len(signal))
    )

    return signal + noise


def channel_frequency_response(taps):
    padded = np.zeros(N_SUBCARRIERS, dtype=complex)

    padded[:len(taps)] = taps

    return np.fft.fft(padded)


# ============================================================
# EQUALIZERS
# ============================================================

def zf_equalizer(received, channel):
    safe_channel = channel.copy()

    safe_channel[np.abs(safe_channel) < 1e-8] = 1e-8

    return received / safe_channel


def mmse_equalizer(received, channel, snr_db):
    snr_linear = 10 ** (snr_db / 10)

    noise_var = 1 / snr_linear

    return (
        np.conj(channel)
        / (np.abs(channel) ** 2 + noise_var)
    ) * received


# ============================================================
# AI DATA
# ============================================================

def make_features(received, channel):
    features = np.column_stack([
        np.real(received),
        np.imag(received),
        np.real(channel),
        np.imag(channel)
    ])

    return features


def make_targets(original):
    return np.column_stack([
        np.real(original),
        np.imag(original)
    ])


def train_ai_model():
    global AI_MODEL
    global INPUT_SCALER
    global OUTPUT_SCALER

    print()
    print("=" * 65)
    print("NEXUS-WAVE AI : TRAINING ADAPTIVE EQUALIZER")
    print("=" * 65)

    X = []
    Y = []

    np.random.seed(42)

    training_frames = 120

    for snr_db in TRAIN_SNR:

        for _ in range(training_frames):

            bits = np.random.randint(
                0,
                2,
                N_SUBCARRIERS * 2
            )

            transmitted_symbols = qpsk_mod(bits)

            tx_signal = ofdm_transmit(
                transmitted_symbols
            )

            taps = multipath_channel()

            rx_signal = apply_channel(
                tx_signal,
                taps
            )

            rx_signal = add_awgn(
                rx_signal,
                snr_db
            )

            received_symbols = ofdm_receive(
                rx_signal
            )

            channel = channel_frequency_response(
                taps
            )

            features = make_features(
                received_symbols,
                channel
            )

            targets = make_targets(
                transmitted_symbols
            )

            X.append(features)
            Y.append(targets)

    X = np.vstack(X)
    Y = np.vstack(Y)

    INPUT_SCALER = StandardScaler()
    OUTPUT_SCALER = StandardScaler()

    X_scaled = INPUT_SCALER.fit_transform(X)
    Y_scaled = OUTPUT_SCALER.fit_transform(Y)

    AI_MODEL = MLPRegressor(
        hidden_layer_sizes=(64, 64),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        batch_size=512,
        max_iter=100,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10
    )

    AI_MODEL.fit(
        X_scaled,
        Y_scaled
    )

    print("AI MODEL TRAINING COMPLETE")
    print("Training samples:", len(X))
    print("=" * 65)
    print()


# ============================================================
# SIMULATION
# ============================================================

def run_simulation(selected_snr=10, frames=30):

    start_time = time.time()

    selected_snr = float(selected_snr)
    frames = int(frames)

    frames = max(5, min(frames, 100))

    ber_zf_total = 0
    ber_mmse_total = 0
    ber_ai_total = 0

    mse_zf_total = 0
    mse_mmse_total = 0
    mse_ai_total = 0

    transmitted_points = []
    received_points = []
    ai_points = []

    channel_mag = None
    channel_phase = None

    for frame in range(frames):

        bits = np.random.randint(
            0,
            2,
            N_SUBCARRIERS * 2
        )

        transmitted_symbols = qpsk_mod(bits)

        tx_signal = ofdm_transmit(
            transmitted_symbols
        )

        taps = multipath_channel()

        rx_signal = apply_channel(
            tx_signal,
            taps
        )

        rx_signal = add_awgn(
            rx_signal,
            selected_snr
        )

        received_symbols = ofdm_receive(
            rx_signal
        )

        channel = channel_frequency_response(
            taps
        )

        # ZF
        zf_symbols = zf_equalizer(
            received_symbols,
            channel
        )

        # MMSE
        mmse_symbols = mmse_equalizer(
            received_symbols,
            channel,
            selected_snr
        )

        # AI
        ai_features = make_features(
            received_symbols,
            channel
        )

        ai_scaled = INPUT_SCALER.transform(
            ai_features
        )

        ai_output_scaled = AI_MODEL.predict(
            ai_scaled
        )

        ai_output = OUTPUT_SCALER.inverse_transform(
            ai_output_scaled
        )

        ai_symbols = (
            ai_output[:, 0]
            + 1j * ai_output[:, 1]
        )

        # Bits
        zf_bits = qpsk_demod(zf_symbols)
        mmse_bits = qpsk_demod(mmse_symbols)
        ai_bits = qpsk_demod(ai_symbols)

        ber_zf_total += np.mean(
            bits != zf_bits
        )

        ber_mmse_total += np.mean(
            bits != mmse_bits
        )

        ber_ai_total += np.mean(
            bits != ai_bits
        )

        mse_zf_total += np.mean(
            np.abs(
                transmitted_symbols - zf_symbols
            ) ** 2
        )

        mse_mmse_total += np.mean(
            np.abs(
                transmitted_symbols - mmse_symbols
            ) ** 2
        )

        mse_ai_total += np.mean(
            np.abs(
                transmitted_symbols - ai_symbols
            ) ** 2
        )

        if frame == frames - 1:

            transmitted_points = [
                {
                    "x": float(np.real(v)),
                    "y": float(np.imag(v))
                }
                for v in transmitted_symbols
            ]

            received_points = [
                {
                    "x": float(np.real(v)),
                    "y": float(np.imag(v))
                }
                for v in received_symbols
            ]

            ai_points = [
                {
                    "x": float(np.real(v)),
                    "y": float(np.imag(v))
                }
                for v in ai_symbols
            ]

            channel_mag = np.abs(channel)
            channel_phase = np.angle(channel)

    ber_zf = ber_zf_total / frames
    ber_mmse = ber_mmse_total / frames
    ber_ai = ber_ai_total / frames

    mse_zf = mse_zf_total / frames
    mse_mmse = mse_mmse_total / frames
    mse_ai = mse_ai_total / frames

    runtime = time.time() - start_time

    return {
        "snr": selected_snr,
        "frames": frames,

        "ber": {
            "zf": float(ber_zf),
            "mmse": float(ber_mmse),
            "ai": float(ber_ai)
        },

        "mse": {
            "zf": float(mse_zf),
            "mmse": float(mse_mmse),
            "ai": float(mse_ai)
        },

        "constellation": {
            "tx": transmitted_points,
            "rx": received_points,
            "ai": ai_points
        },

        "channel": {
            "magnitude": channel_mag.tolist(),
            "phase": channel_phase.tolist()
        },

        "runtime": round(runtime, 3),

        "parameters": {
            "subcarriers": N_SUBCARRIERS,
            "cyclic_prefix": CP_LENGTH,
            "paths": 3,
            "path_gains": PATH_GAINS.tolist(),
            "modulation": "QPSK",
            "equalizers": [
                "ZF",
                "MMSE",
                "AI-MLP"
            ]
        }
    }


# ============================================================
# LOGIN HTML
# ============================================================

LOGIN_HTML = """
<!DOCTYPE html>
<html>
<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>NEXUS-WAVE AI | Login</title>

<style>

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    min-height: 100vh;
    overflow: hidden;
    font-family: Arial, Helvetica, sans-serif;

    background:
        radial-gradient(circle at 20% 20%, rgba(0,255,255,.12), transparent 30%),
        radial-gradient(circle at 80% 70%, rgba(170,0,255,.18), transparent 35%),
        linear-gradient(135deg, #020617, #070022, #00152d);

    color: white;
}

canvas {
    position: fixed;
    inset: 0;
    width: 100%;
    height: 100%;
    z-index: 0;
}

.login-wrapper {
    position: relative;
    z-index: 2;

    width: 100%;
    min-height: 100vh;

    display: flex;
    align-items: center;
    justify-content: center;

    padding: 25px;
}

.login-card {
    width: 440px;
    max-width: 100%;

    padding: 42px;

    border: 1px solid rgba(0,255,255,.25);

    border-radius: 28px;

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,.12),
            rgba(255,255,255,.025)
        );

    backdrop-filter: blur(25px);

    box-shadow:
        0 0 50px rgba(0,255,255,.12),
        0 0 100px rgba(150,0,255,.10);

    animation: floatCard 5s ease-in-out infinite;
}

@keyframes floatCard {

    0%,100% {
        transform: translateY(0);
    }

    50% {
        transform: translateY(-10px);
    }
}

.logo {
    width: 90px;
    height: 90px;

    margin: auto;

    border-radius: 50%;

    display: flex;
    align-items: center;
    justify-content: center;

    font-size: 38px;

    background:
        linear-gradient(
            135deg,
            #00f5ff,
            #7c3aed,
            #ec4899
        );

    box-shadow:
        0 0 30px rgba(0,255,255,.5),
        0 0 60px rgba(168,85,247,.3);

    animation: spinGlow 5s linear infinite;
}

@keyframes spinGlow {

    0% {
        transform: rotate(0deg);
    }

    100% {
        transform: rotate(360deg);
    }
}

h1 {
    text-align: center;
    margin-top: 24px;

    font-size: 30px;
    letter-spacing: 2px;

    background:
        linear-gradient(
            90deg,
            #00f5ff,
            #a855f7,
            #f472b6
        );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    margin-top: 10px;

    color: #a5b4fc;
    font-size: 13px;
}

.input-group {
    margin-top: 28px;
}

label {
    display: block;
    margin-bottom: 8px;

    color: #cbd5e1;
    font-size: 13px;
}

input {
    width: 100%;

    padding: 15px 17px;

    border-radius: 14px;

    border: 1px solid rgba(255,255,255,.12);

    outline: none;

    background: rgba(0,0,0,.3);

    color: white;

    font-size: 15px;

    transition: .3s;
}

input:focus {
    border-color: #00f5ff;

    box-shadow:
        0 0 20px rgba(0,245,255,.18);
}

button {
    width: 100%;

    margin-top: 25px;

    padding: 15px;

    border: none;

    border-radius: 14px;

    color: white;

    font-weight: bold;
    font-size: 15px;

    cursor: pointer;

    background:
        linear-gradient(
            90deg,
            #0891b2,
            #7c3aed,
            #db2777
        );

    box-shadow:
        0 0 25px rgba(124,58,237,.35);

    transition: .3s;
}

button:hover {
    transform: translateY(-2px);

    box-shadow:
        0 0 35px rgba(0,245,255,.35);
}

.error {
    margin-top: 15px;

    text-align: center;

    color: #fb7185;

    font-size: 13px;
}

.demo {
    margin-top: 22px;

    padding: 12px;

    border-radius: 12px;

    text-align: center;

    font-size: 12px;

    color: #94a3b8;

    background: rgba(255,255,255,.04);
}

</style>

</head>

<body>

<canvas id="space"></canvas>

<div class="login-wrapper">

    <div class="login-card">

        <div class="logo">∿</div>

        <h1>NEXUS-WAVE AI</h1>

        <div class="subtitle">
            Intelligent 3D OFDM Communication Laboratory
        </div>

        <form method="POST" action="/login">

            <div class="input-group">

                <label>USERNAME</label>

                <input
                    type="text"
                    name="username"
                    placeholder="Enter username"
                    required
                >

            </div>

            <div class="input-group">

                <label>PASSWORD</label>

                <input
                    type="password"
                    name="password"
                    placeholder="Enter password"
                    required
                >

            </div>

            <button type="submit">
                ENTER COMMUNICATION LAB
            </button>

        </form>

        {% if error %}

        <div class="error">
            {{ error }}
        </div>

        {% endif %}

        <div class="demo">
            Demo access: admin / ofdm2026
        </div>

    </div>

</div>


<script>

const canvas = document.getElementById("space");
const ctx = canvas.getContext("2d");

let particles = [];

function resize() {

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    particles = [];

    for (let i = 0; i < 140; i++) {

        particles.push({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            r: Math.random() * 2 + .4,
            dx: (Math.random() - .5) * .5,
            dy: (Math.random() - .5) * .5
        });

    }
}

resize();

window.addEventListener("resize", resize);

function animate() {

    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );

    for (const p of particles) {

        p.x += p.dx;
        p.y += p.dy;

        if (p.x < 0 || p.x > canvas.width) {
            p.dx *= -1;
        }

        if (p.y < 0 || p.y > canvas.height) {
            p.dy *= -1;
        }

        ctx.beginPath();

        ctx.arc(
            p.x,
            p.y,
            p.r,
            0,
            Math.PI * 2
        );

        ctx.fillStyle = "rgba(0,245,255,.65)";

        ctx.fill();
    }

    requestAnimationFrame(animate);
}

animate();

</script>

</body>
</html>
"""


# ============================================================
# DASHBOARD HTML
# ============================================================

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>NEXUS-WAVE AI Dashboard</title>

<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>

<style>

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {

    min-height: 100vh;

    font-family:
        Arial,
        Helvetica,
        sans-serif;

    color: #f8fafc;

    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(0,245,255,.12),
            transparent 25%
        ),

        radial-gradient(
            circle at 90% 80%,
            rgba(217,70,239,.14),
            transparent 30%
        ),

        linear-gradient(
            135deg,
            #020617,
            #080020,
            #00182c
        );

    overflow-x: hidden;
}

canvas#particles {
    position: fixed;

    inset: 0;

    width: 100%;
    height: 100%;

    pointer-events: none;

    z-index: 0;
}

.app {

    position: relative;

    z-index: 2;

    min-height: 100vh;

    display: flex;
}

.sidebar {

    width: 255px;

    min-height: 100vh;

    padding: 22px 15px;

    position: fixed;

    left: 0;
    top: 0;
    bottom: 0;

    border-right:
        1px solid
        rgba(255,255,255,.08);

    background:
        rgba(2,6,23,.78);

    backdrop-filter:
        blur(25px);

    z-index: 10;
}

.brand {

    padding: 15px;

    border-radius: 18px;

    background:
        linear-gradient(
            135deg,
            rgba(0,245,255,.10),
            rgba(168,85,247,.12)
        );

    border:
        1px solid
        rgba(0,245,255,.15);

    margin-bottom: 25px;
}

.brand h2 {

    font-size: 18px;

    letter-spacing: 1px;

    background:
        linear-gradient(
            90deg,
            #00f5ff,
            #a855f7,
            #f472b6
        );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;
}

.brand p {

    color: #94a3b8;

    font-size: 10px;

    margin-top: 6px;

    line-height: 1.5;
}

.nav-title {

    color: #64748b;

    font-size: 10px;

    letter-spacing: 2px;

    margin: 20px 10px 8px;
}

.nav-btn {

    width: 100%;

    border: none;

    background: transparent;

    color: #94a3b8;

    text-align: left;

    padding: 12px 13px;

    margin: 3px 0;

    border-radius: 12px;

    cursor: pointer;

    transition: .25s;

    font-size: 13px;
}

.nav-btn:hover,
.nav-btn.active {

    color: white;

    background:
        linear-gradient(
            90deg,
            rgba(0,245,255,.12),
            rgba(124,58,237,.18)
        );

    box-shadow:
        inset 3px 0 0 #00f5ff;
}

.logout {

    position: absolute;

    left: 15px;
    right: 15px;
    bottom: 20px;

    text-align: center;

    padding: 12px;

    border-radius: 12px;

    text-decoration: none;

    color: #fda4af;

    background:
        rgba(244,63,94,.08);

    border:
        1px solid
        rgba(244,63,94,.15);

    font-size: 13px;
}

.main {

    margin-left: 255px;

    width: calc(100% - 255px);

    padding: 24px;

    min-height: 100vh;
}

.topbar {

    display: flex;

    justify-content: space-between;

    align-items: center;

    gap: 15px;

    margin-bottom: 25px;
}

.page-title h1 {

    font-size: 28px;

    letter-spacing: .5px;
}

.page-title p {

    color: #94a3b8;

    font-size: 12px;

    margin-top: 5px;
}

.status {

    display: flex;

    align-items: center;

    gap: 8px;

    padding: 9px 13px;

    border-radius: 30px;

    border:
        1px solid
        rgba(34,197,94,.25);

    background:
        rgba(34,197,94,.07);

    color: #86efac;

    font-size: 12px;
}

.status-dot {

    width: 8px;
    height: 8px;

    border-radius: 50%;

    background: #22c55e;

    box-shadow:
        0 0 12px #22c55e;

    animation: pulse 1.5s infinite;
}

@keyframes pulse {

    0%,100% {
        opacity: 1;
    }

    50% {
        opacity: .35;
    }
}

.page {

    display: none;

    animation:
        pageIn .45s ease;
}

.page.active {
    display: block;
}

@keyframes pageIn {

    from {
        opacity: 0;
        transform: translateY(12px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.cards {

    display: grid;

    grid-template-columns:
        repeat(
            4,
            minmax(0,1fr)
        );

    gap: 15px;

    margin-bottom: 18px;
}

.card {

    padding: 18px;

    border-radius: 18px;

    border:
        1px solid
        rgba(255,255,255,.08);

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,.075),
            rgba(255,255,255,.025)
        );

    backdrop-filter:
        blur(18px);

    box-shadow:
        0 15px 40px
        rgba(0,0,0,.18);

    transition: .3s;
}

.card:hover {

    transform:
        translateY(-4px);

    border-color:
        rgba(0,245,255,.20);

    box-shadow:
        0 15px 45px
        rgba(0,245,255,.08);
}

.metric-label {

    color: #94a3b8;

    font-size: 11px;

    letter-spacing: 1px;

    text-transform: uppercase;
}

.metric-value {

    font-size: 28px;

    font-weight: bold;

    margin-top: 9px;

    background:
        linear-gradient(
            90deg,
            #67e8f9,
            #c084fc,
            #f9a8d4
        );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;
}

.metric-small {

    color: #64748b;

    font-size: 10px;

    margin-top: 6px;
}

.grid {

    display: grid;

    grid-template-columns:
        repeat(
            2,
            minmax(0,1fr)
        );

    gap: 18px;
}

.panel {

    padding: 18px;

    border-radius: 20px;

    border:
        1px solid
        rgba(255,255,255,.08);

    background:
        rgba(255,255,255,.045);

    backdrop-filter:
        blur(18px);

    min-width: 0;
}

.panel-title {

    display: flex;

    justify-content: space-between;

    align-items: center;

    margin-bottom: 15px;
}

.panel-title h3 {

    font-size: 14px;

    letter-spacing: .5px;
}

.panel-title span {

    color: #64748b;

    font-size: 10px;
}

.chart {

    width: 100%;

    height: 350px;
}

.controls {

    display: grid;

    grid-template-columns:
        repeat(
            2,
            minmax(0,1fr)
        );

    gap: 15px;

    margin-bottom: 18px;
}

.control {

    padding: 18px;

    border-radius: 18px;

    border:
        1px solid
        rgba(255,255,255,.08);

    background:
        rgba(255,255,255,.045);
}

.control label {

    display: block;

    font-size: 11px;

    color: #94a3b8;

    margin-bottom: 8px;
}

select,
input[type=number] {

    width: 100%;

    padding: 12px;

    color: white;

    background: #0f172a;

    border:
        1px solid
        rgba(255,255,255,.12);

    border-radius: 10px;

    outline: none;
}

.run-btn {

    width: 100%;

    padding: 13px;

    margin-top: 10px;

    border: none;

    border-radius: 11px;

    color: white;

    font-weight: bold;

    cursor: pointer;

    background:
        linear-gradient(
            90deg,
            #0891b2,
            #7c3aed,
            #db2777
        );

    transition: .25s;
}

.run-btn:hover {

    transform: translateY(-2px);

    box-shadow:
        0 0 30px
        rgba(124,58,237,.3);
}

.architecture {

    display: grid;

    grid-template-columns:
        repeat(
            5,
            minmax(0,1fr)
        );

    gap: 10px;

    align-items: center;
}

.arch-box {

    min-height: 120px;

    padding: 15px;

    display: flex;

    align-items: center;

    justify-content: center;

    flex-direction: column;

    text-align: center;

    border-radius: 18px;

    border:
        1px solid
        rgba(0,245,255,.16);

    background:
        linear-gradient(
            145deg,
            rgba(0,245,255,.08),
            rgba(124,58,237,.08)
        );

    position: relative;

    overflow: hidden;
}

.arch-box::before {

    content: "";

    position: absolute;

    width: 100px;
    height: 100px;

    border-radius: 50%;

    border:
        1px solid
        rgba(0,245,255,.20);

    animation:
        rotateRing 5s linear infinite;
}

@keyframes rotateRing {

    from {
        transform: rotate(0deg);
    }

    to {
        transform: rotate(360deg);
    }
}

.arch-icon {

    font-size: 28px;

    margin-bottom: 8px;

    position: relative;

    z-index: 2;
}

.arch-name {

    font-weight: bold;

    font-size: 12px;

    position: relative;

    z-index: 2;
}

.arch-desc {

    font-size: 9px;

    color: #94a3b8;

    margin-top: 5px;

    position: relative;

    z-index: 2;
}

.arrow {

    text-align: center;

    font-size: 22px;

    color: #22d3ee;

}

.parameter-table {

    width: 100%;

    border-collapse: collapse;
}

.parameter-table tr {

    border-bottom:
        1px solid
        rgba(255,255,255,.06);
}

.parameter-table td {

    padding: 13px 8px;

    font-size: 12px;
}

.parameter-table td:first-child {

    color: #94a3b8;
}

.parameter-table td:last-child {

    color: #67e8f9;

    text-align: right;

    font-weight: bold;
}

.ai-box {

    position: relative;

    min-height: 350px;

    overflow: hidden;
}

#neuralCanvas {

    width: 100%;
    height: 350px;
}

.info-box {

    padding: 20px;

    border-radius: 18px;

    border:
        1px solid
        rgba(168,85,247,.18);

    background:
        linear-gradient(
            135deg,
            rgba(124,58,237,.08),
            rgba(236,72,153,.06)
        );

    line-height: 1.8;

    color: #cbd5e1;

    font-size: 13px;
}

.info-box strong {

    color: #67e8f9;
}

.badges {

    display: flex;

    flex-wrap: wrap;

    gap: 8px;

    margin-top: 15px;
}

.badge {

    padding: 7px 10px;

    border-radius: 20px;

    background:
        rgba(0,245,255,.07);

    border:
        1px solid
        rgba(0,245,255,.15);

    color: #67e8f9;

    font-size: 10px;
}

@media(max-width: 1100px) {

    .cards {
        grid-template-columns:
            repeat(2,1fr);
    }

    .architecture {
        grid-template-columns:
            repeat(2,1fr);
    }

    .arrow {
        display: none;
    }
}

@media(max-width: 800px) {

    .sidebar {

        width: 70px;

        padding: 12px 8px;
    }

    .brand h2,
    .brand p,
    .nav-title,
    .nav-btn span,
    .logout span {

        display: none;
    }

    .brand {

        text-align: center;

        padding: 12px 5px;
    }

    .nav-btn {

        text-align: center;
    }

    .main {

        margin-left: 70px;

        width:
            calc(
                100% - 70px
            );

        padding: 15px;
    }

    .cards,
    .grid,
    .controls {

        grid-template-columns:
            1fr;
    }

    .topbar {

        flex-direction: column;

        align-items: flex-start;
    }
}

</style>

</head>

<body>

<canvas id="particles"></canvas>

<div class="app">

<!-- ====================================================== -->
<!-- SIDEBAR -->
<!-- ====================================================== -->

<aside class="sidebar">

    <div class="brand">

        <h2>NEXUS-WAVE AI</h2>

        <p>
            Next-Generation Intelligent
            Communication Laboratory
        </p>

    </div>

    <div class="nav-title">
        NAVIGATION
    </div>

    <button class="nav-btn active"
            onclick="showPage('home', this)">
        ◈ <span>Home</span>
    </button>

    <button class="nav-btn"
            onclick="showPage('performance', this)">
        ◉ <span>Performance</span>
    </button>

    <button class="nav-btn"
            onclick="showPage('channel', this)">
        ≋ <span>Channel Analysis</span>
    </button>

    <button class="nav-btn"
            onclick="showPage('ai', this)">
        ◎ <span>AI Equalizer</span>
    </button>

    <button class="nav-btn"
            onclick="showPage('analytics', this)">
        ◌ <span>BER / MSE Analytics</span>
    </button>

    <button class="nav-btn"
            onclick="showPage('architecture', this)">
        ◇ <span>Architecture</span>
    </button>

    <button class="nav-btn"
            onclick="showPage('parameters', this)">
        ⚙ <span>Parameters</span>
    </button>

    <a href="/logout" class="logout">
        ⇥ <span>Logout</span>
    </a>

</aside>


<!-- ====================================================== -->
<!-- MAIN -->
<!-- ====================================================== -->

<main class="main">

<div class="topbar">

    <div class="page-title">

        <h1 id="pageHeading">
            Communication Command Center
        </h1>

        <p id="pageSubheading">
            Intelligent OFDM transmission,
            channel analysis and AI equalization
        </p>

    </div>

    <div class="status">

        <div class="status-dot"></div>

        SYSTEM ONLINE

    </div>

</div>


<!-- ====================================================== -->
<!-- HOME -->
<!-- ====================================================== -->

<section id="home" class="page active">

<div class="cards">

    <div class="card">

        <div class="metric-label">
            Subcarriers
        </div>

        <div class="metric-value"
             data-target="64">
            0
        </div>

        <div class="metric-small">
            OFDM spectral channels
        </div>

    </div>

    <div class="card">

        <div class="metric-label">
            CP Length
        </div>

        <div class="metric-value"
             data-target="16">
            0
        </div>

        <div class="metric-small">
            Cyclic prefix samples
        </div>

    </div>

    <div class="card">

        <div class="metric-label">
            AI Layers
        </div>

        <div class="metric-value"
             data-target="2">
            0
        </div>

        <div class="metric-small">
            Neural network hidden layers
        </div>

    </div>

    <div class="card">

        <div class="metric-label">
            Channel Paths
        </div>

        <div class="metric-value"
             data-target="3">
            0
        </div>

        <div class="metric-small">
            Multipath fading taps
        </div>

    </div>

</div>


<div class="grid">

    <div class="panel">

        <div class="panel-title">

            <h3>
                3D OFDM Signal Space
            </h3>

            <span>
                LIVE VISUALIZATION
            </span>

        </div>

        <div id="ofdm3d"
             class="chart">
        </div>

    </div>


    <div class="panel">

        <div class="panel-title">

            <h3>
                System Overview
            </h3>

            <span>
                NEXUS-WAVE
            </span>

        </div>

        <div class="info-box">

            <strong>NEXUS-WAVE AI</strong>
            is an intelligent OFDM communication
            laboratory that combines classical
            equalization techniques with an
            adaptive neural-network equalizer.

            <br><br>

            The system simulates QPSK transmission,
            64-subcarrier OFDM, cyclic prefix,
            multipath fading and AWGN.

            <br><br>

            Three equalization approaches are
            evaluated:

            <div class="badges">

                <div class="badge">ZF</div>
                <div class="badge">MMSE</div>
                <div class="badge">AI-MLP</div>
                <div class="badge">QPSK</div>
                <div class="badge">OFDM</div>
                <div class="badge">AWGN</div>
                <div class="badge">MULTIPATH</div>

            </div>

        </div>

    </div>

</div>

</section>


<!-- ====================================================== -->
<!-- PERFORMANCE -->
<!-- ====================================================== -->

<section id="performance" class="page">

<div class="controls">

    <div class="control">

        <label>
            TEST SNR
        </label>

        <select id="snrSelect">

            <option value="-5">-5 dB</option>
            <option value="0">0 dB</option>
            <option value="5">5 dB</option>
            <option value="10" selected>10 dB</option>
            <option value="15">15 dB</option>
            <option value="20">20 dB</option>
            <option value="25">25 dB</option>

        </select>

    </div>


    <div class="control">

        <label>
            SIMULATION FRAMES
        </label>

        <input
            id="frameInput"
            type="number"
            min="5"
            max="100"
            value="30"
        >

        <button
            class="run-btn"
            onclick="runSimulation()">

            RUN COMMUNICATION SIMULATION

        </button>

    </div>

</div>


<div class="cards">

    <div class="card">

        <div class="metric-label">
            ZF BER
        </div>

        <div class="metric-value"
             id="zfBer">
            —
        </div>

        <div class="metric-small">
            Zero Forcing
        </div>

    </div>

    <div class="card">

        <div class="metric-label">
            MMSE BER
        </div>

        <div class="metric-value"
             id="mmseBer">
            —
        </div>

        <div class="metric-small">
            Minimum Mean Square Error
        </div>

    </div>

    <div class="card">

        <div class="metric-label">
            AI BER
        </div>

        <div class="metric-value"
             id="aiBer">
            —
        </div>

        <div class="metric-small">
            Adaptive MLP Equalizer
        </div>

    </div>

    <div class="card">

        <div class="metric-label">
            Runtime
        </div>

        <div class="metric-value"
             id="runtime">
            —
        </div>

        <div class="metric-small">
            Seconds
        </div>

    </div>

</div>


<div class="grid">

    <div class="panel">

        <div class="panel-title">

            <h3>
                BER Comparison
            </h3>

            <span>
                LOG SCALE
            </span>

        </div>

        <div id="berChart"
             class="chart">
        </div>

    </div>


    <div class="panel">

        <div class="panel-title">

            <h3>
                MSE Comparison
            </h3>

            <span>
                ERROR POWER
            </span>

        </div>

        <div id="mseChart"
             class="chart">
        </div>

    </div>

</div>

</section>


<!-- ====================================================== -->
<!-- CHANNEL -->
<!-- ====================================================== -->

<section id="channel" class="page">

<div class="grid">

    <div class="panel">

        <div class="panel-title">

            <h3>
                Channel Magnitude Response
            </h3>

            <span>
                64 SUBCARRIERS
            </span>

        </div>

        <div id="channelMagnitude"
             class="chart">
        </div>

    </div>


    <div class="panel">

        <div class="panel-title">

            <h3>
                Channel Phase Response
            </h3>

            <span>
                RADIANS
            </span>

        </div>

        <div id="channelPhase"
             class="chart">
        </div>

    </div>

</div>


<div class="panel"
     style="margin-top:18px">

    <div class="panel-title">

        <h3>
            Multipath Channel Model
        </h3>

        <span>
            3-PATH FADING
        </span>

    </div>

    <div class="cards">

        <div class="card">

            <div class="metric-label">
                Path 1
            </div>

            <div class="metric-value">
                0.90
            </div>

            <div class="metric-small">
                Dominant path
            </div>

        </div>

        <div class="card">

            <div class="metric-label">
                Path 2
            </div>

            <div class="metric-value">
                0.40
            </div>

            <div class="metric-small">
                Delayed path
            </div>

        </div>

        <div class="card">

            <div class="metric-label">
                Path 3
            </div>

            <div class="metric-value">
                0.20
            </div>

            <div class="metric-small">
                Delayed path
            </div>

        </div>

        <div class="card">

            <div class="metric-label">
                Noise
            </div>

            <div class="metric-value">
                AWGN
            </div>

            <div class="metric-small">
                SNR-controlled
            </div>

        </div>

    </div>

</div>

</section>


<!-- ====================================================== -->
<!-- AI EQUALIZER -->
<!-- ====================================================== -->

<section id="ai" class="page">

<div class="grid">

    <div class="panel ai-box">

        <div class="panel-title">

            <h3>
                AI Neural Equalizer
            </h3>

            <span>
                MLP • 64 → 64
            </span>

        </div>

        <canvas id="neuralCanvas">
        </canvas>

    </div>


    <div class="panel">

        <div class="panel-title">

            <h3>
                AI Equalizer Pipeline
            </h3>

            <span>
                ADAPTIVE PROCESS
            </span>

        </div>

        <div class="info-box">

            <strong>Input:</strong>
            Received complex OFDM symbols
            and channel response.

            <br><br>

            <strong>Pre-processing:</strong>
            StandardScaler normalization.

            <br><br>

            <strong>Neural Network:</strong>
            MLPRegressor with two hidden
            layers of 64 neurons.

            <br><br>

            <strong>Output:</strong>
            Reconstructed real and imaginary
            components of transmitted symbols.

            <br><br>

            <strong>Decision:</strong>
            QPSK demodulation produces the
            recovered bit stream.

        </div>

    </div>

</div>


<div class="panel"
     style="margin-top:18px">

    <div class="panel-title">

        <h3>
            AI Recovered Constellation
        </h3>

        <span>
            REAL vs IMAGINARY
        </span>

    </div>

    <div id="aiConstellation"
         class="chart">
    </div>

</div>

</section>


<!-- ====================================================== -->
<!-- ANALYTICS -->
<!-- ====================================================== -->

<section id="analytics" class="page">

<div class="grid">

    <div class="panel">

        <div class="panel-title">

            <h3>
                Transmitted Constellation
            </h3>

            <span>
                QPSK
            </span>

        </div>

        <div id="txConstellation"
             class="chart">
        </div>

    </div>


    <div class="panel">

        <div class="panel-title">

            <h3>
                Received Constellation
            </h3>

            <span>
                CHANNEL OUTPUT
            </span>

        </div>

        <div id="rxConstellation"
             class="chart">
        </div>

    </div>

</div>

</section>


<!-- ====================================================== -->
<!-- ARCHITECTURE -->
<!-- ====================================================== -->

<section id="architecture" class="page">

<div class="panel">

    <div class="panel-title">

        <h3>
            NEXUS-WAVE AI Communication Architecture
        </h3>

        <span>
            END-TO-END
        </span>

    </div>


    <div class="architecture">

        <div class="arch-box">

            <div class="arch-icon">⌁</div>

            <div class="arch-name">
                BIT SOURCE
            </div>

            <div class="arch-desc">
                Random binary data
            </div>

        </div>

        <div class="arrow">→</div>

        <div class="arch-box">

            <div class="arch-icon">✦</div>

            <div class="arch-name">
                QPSK
            </div>

            <div class="arch-desc">
                Digital modulation
            </div>

        </div>

        <div class="arrow">→</div>

        <div class="arch-box">

            <div class="arch-icon">▦</div>

            <div class="arch-name">
                OFDM
            </div>

            <div class="arch-desc">
                64 subcarriers
            </div>

        </div>

        <div class="arrow">→</div>

        <div class="arch-box">

            <div class="arch-icon">≈</div>

            <div class="arch-name">
                CHANNEL
            </div>

            <div class="arch-desc">
                Multipath + AWGN
            </div>

        </div>

    </div>


    <div style="height:25px">
    </div>


    <div class="architecture">

        <div class="arch-box">

            <div class="arch-icon">◈</div>

            <div class="arch-name">
                ZF
            </div>

            <div class="arch-desc">
                Classical equalizer
            </div>

        </div>

        <div class="arrow">→</div>

        <div class="arch-box">

            <div class="arch-icon">◉</div>

            <div class="arch-name">
                MMSE
            </div>

            <div class="arch-desc">
                Noise-aware equalizer
            </div>

        </div>

        <div class="arrow">→</div>

        <div class="arch-box">

            <div class="arch-icon">◎</div>

            <div class="arch-name">
                AI MLP
            </div>

            <div class="arch-desc">
                Adaptive equalization
            </div>

        </div>

        <div class="arrow">→</div>

        <div class="arch-box">

            <div class="arch-icon">✓</div>

            <div class="arch-name">
                QPSK DEMOD
            </div>

            <div class="arch-desc">
                Recovered bits
            </div>

        </div>

    </div>

</div>

</section>


<!-- ====================================================== -->
<!-- PARAMETERS -->
<!-- ====================================================== -->

<section id="parameters" class="page">

<div class="grid">

    <div class="panel">

        <div class="panel-title">

            <h3>
                OFDM Parameters
            </h3>

        </div>

        <table class="parameter-table">

            <tr>
                <td>Modulation</td>
                <td>QPSK</td>
            </tr>

            <tr>
                <td>Subcarriers</td>
                <td>64</td>
            </tr>

            <tr>
                <td>Cyclic Prefix</td>
                <td>16</td>
            </tr>

            <tr>
                <td>Channel</td>
                <td>3-Path Multipath</td>
            </tr>

            <tr>
                <td>Path Gains</td>
                <td>0.9 / 0.4 / 0.2</td>
            </tr>

            <tr>
                <td>Noise</td>
                <td>AWGN</td>
            </tr>

        </table>

    </div>


    <div class="panel">

        <div class="panel-title">

            <h3>
                AI Parameters
            </h3>

        </div>

        <table class="parameter-table">

            <tr>
                <td>Model</td>
                <td>MLPRegressor</td>
            </tr>

            <tr>
                <td>Hidden Layer 1</td>
                <td>64 neurons</td>
            </tr>

            <tr>
                <td>Hidden Layer 2</td>
                <td>64 neurons</td>
            </tr>

            <tr>
                <td>Activation</td>
                <td>ReLU</td>
            </tr>

            <tr>
                <td>Optimizer</td>
                <td>Adam</td>
            </tr>

            <tr>
                <td>Learning Rate</td>
                <td>0.001</td>
            </tr>

            <tr>
                <td>Scaler</td>
                <td>StandardScaler</td>
            </tr>

        </table>

    </div>

</div>


<div class="panel"
     style="margin-top:18px">

    <div class="panel-title">

        <h3>
            Project Information
        </h3>

    </div>

    <div class="info-box">

        <strong>Project:</strong>
        NEXUS-WAVE AI

        <br>

        <strong>Full Title:</strong>
        Intelligent 3D OFDM Communication
        & Adaptive Channel Equalization Platform

        <br>

        <strong>Domain:</strong>
        Electronics & Communication Engineering

        <br>

        <strong>Technology:</strong>
        Python + Flask + NumPy +
        Scikit-learn + Plotly

        <br>

        <strong>Communication Techniques:</strong>
        QPSK, OFDM, Multipath Fading,
        AWGN, ZF, MMSE and AI Equalization

    </div>

</div>

</section>


</main>

</div>


<script>

// ============================================================
// PARTICLE BACKGROUND
// ============================================================

const particleCanvas =
    document.getElementById("particles");

const pctx =
    particleCanvas.getContext("2d");

let stars = [];

function resizeParticles() {

    particleCanvas.width =
        window.innerWidth;

    particleCanvas.height =
        window.innerHeight;

    stars = [];

    for (let i = 0; i < 150; i++) {

        stars.push({

            x: Math.random() *
                particleCanvas.width,

            y: Math.random() *
                particleCanvas.height,

            r: Math.random() * 1.7 + .3,

            dx:
                (Math.random() - .5) *
                .4,

            dy:
                (Math.random() - .5) *
                .4
        });
    }
}

resizeParticles();

window.addEventListener(
    "resize",
    resizeParticles
);

function animateParticles() {

    pctx.clearRect(
        0,
        0,
        particleCanvas.width,
        particleCanvas.height
    );

    for (const s of stars) {

        s.x += s.dx;
        s.y += s.dy;

        if (
            s.x < 0 ||
            s.x > particleCanvas.width
        ) {
            s.dx *= -1;
        }

        if (
            s.y < 0 ||
            s.y > particleCanvas.height
        ) {
            s.dy *= -1;
        }

        pctx.beginPath();

        pctx.arc(
            s.x,
            s.y,
            s.r,
            0,
            Math.PI * 2
        );

        pctx.fillStyle =
            "rgba(0,245,255,.45)";

        pctx.fill();
    }

    requestAnimationFrame(
        animateParticles
    );
}

animateParticles();


// ============================================================
// PAGE NAVIGATION
// ============================================================

const headings = {

    home: [
        "Communication Command Center",
        "Intelligent OFDM transmission, channel analysis and AI equalization"
    ],

    performance: [
        "Performance Laboratory",
        "Compare ZF, MMSE and AI equalization in real-time simulation"
    ],

    channel: [
        "Channel Analysis",
        "Visualize multipath fading and frequency-domain channel response"
    ],

    ai: [
        "AI Equalizer Laboratory",
        "Adaptive neural-network based OFDM symbol recovery"
    ],

    analytics: [
        "BER / MSE Analytics",
        "Analyze transmitted, received and recovered signal quality"
    ],

    architecture: [
        "System Architecture",
        "End-to-end NEXUS-WAVE AI communication pipeline"
    ],

    parameters: [
        "System Parameters",
        "OFDM, channel and AI configuration"
    ]
};


function showPage(pageId, button) {

    document
        .querySelectorAll(".page")
        .forEach(page => {

            page.classList.remove(
                "active"
            );
        });

    document
        .querySelectorAll(".nav-btn")
        .forEach(btn => {

            btn.classList.remove(
                "active"
            );
        });

    document
        .getElementById(pageId)
        .classList.add("active");

    if (button) {
        button.classList.add("active");
    }

    document.getElementById(
        "pageHeading"
    ).textContent =
        headings[pageId][0];

    document.getElementById(
        "pageSubheading"
    ).textContent =
        headings[pageId][1];

    if (pageId === "home") {
        draw3D();
    }

    if (pageId === "ai") {
        drawNeural();
    }
}


// ============================================================
// PLOTLY COMMON
// ============================================================

const plotLayout = {

    paper_bgcolor:
        "rgba(0,0,0,0)",

    plot_bgcolor:
        "rgba(0,0,0,0)",

    font: {
        color: "#cbd5e1"
    },

    margin: {
        l: 45,
        r: 20,
        t: 15,
        b: 45
    },

    xaxis: {
        gridcolor:
            "rgba(148,163,184,.10)"
    },

    yaxis: {
        gridcolor:
            "rgba(148,163,184,.10)"
    },

    showlegend: true,

    legend: {
        orientation: "h",
        y: -0.18
    }
};


// ============================================================
// 3D OFDM
// ============================================================

function draw3D() {

    const x = [];
    const y = [];
    const z = [];

    for (let i = 0; i < 64; i++) {

        const angle =
            i * Math.PI * 2 / 16;

        x.push(
            Math.cos(angle) *
            (1 + i / 100)
        );

        y.push(
            Math.sin(angle) *
            (1 + i / 100)
        );

        z.push(
            Math.sin(i / 5) * .6
        );
    }

    Plotly.newPlot(
        "ofdm3d",
        [{
            x: x,
            y: y,
            z: z,

            type: "scatter3d",

            mode: "lines+markers",

            marker: {
                size: 4,
                color: z,
                colorscale: "Plasma"
            },

            line: {
                width: 3,
                color: "#00f5ff"
            }
        }],
        {
            ...plotLayout,

            scene: {

                bgcolor:
                    "rgba(0,0,0,0)",

                xaxis: {
                    title: "Real",
                    gridcolor:
                        "rgba(0,245,255,.08)"
                },

                yaxis: {
                    title: "Imaginary",
                    gridcolor:
                        "rgba(0,245,255,.08)"
                },

                zaxis: {
                    title: "Subcarrier",
                    gridcolor:
                        "rgba(0,245,255,.08)"
                }
            },

            margin: {
                l: 0,
                r: 0,
                t: 0,
                b: 0
            }
        },
        {
            responsive: true,
            displayModeBar: false
        }
    );
}

draw3D();


// ============================================================
// SIMULATION
// ============================================================

async function runSimulation() {

    const snr =
        document.getElementById(
            "snrSelect"
        ).value;

    const frames =
        document.getElementById(
            "frameInput"
        ).value;

    document.getElementById(
        "zfBer"
    ).textContent = "...";

    document.getElementById(
        "mmseBer"
    ).textContent = "...";

    document.getElementById(
        "aiBer"
    ).textContent = "...";

    try {

        const response =
            await fetch(
                `/simulate?snr=${snr}&frames=${frames}`
            );

        const data =
            await response.json();

        if (!response.ok) {
            throw new Error(
                data.error || "Simulation failed"
            );
        }

        updateDashboard(data);

    } catch (error) {

        alert(
            "Simulation error: " +
            error.message
        );
    }
}


// ============================================================
// UPDATE DASHBOARD
// ============================================================

function updateDashboard(data) {

    document.getElementById(
        "zfBer"
    ).textContent =
        data.ber.zf.toExponential(3);

    document.getElementById(
        "mmseBer"
    ).textContent =
        data.ber.mmse.toExponential(3);

    document.getElementById(
        "aiBer"
    ).textContent =
        data.ber.ai.toExponential(3);

    document.getElementById(
        "runtime"
    ).textContent =
        data.runtime.toFixed(2) + "s";


    // BER chart

    Plotly.newPlot(
        "berChart",

        [{
            x: [
                "ZF",
                "MMSE",
                "AI-MLP"
            ],

            y: [
                data.ber.zf,
                data.ber.mmse,
                data.ber.ai
            ],

            type: "bar",

            marker: {
                color: [
                    "#06b6d4",
                    "#8b5cf6",
                    "#ec4899"
                ]
            }
        }],

        {
            ...plotLayout,

            yaxis: {
                type: "log",
                title: "BER",
                gridcolor:
                    "rgba(148,163,184,.10)"
            }
        },

        {
            responsive: true,
            displayModeBar: false
        }
    );


    // MSE chart

    Plotly.newPlot(
        "mseChart",

        [{
            x: [
                "ZF",
                "MMSE",
                "AI-MLP"
            ],

            y: [
                data.mse.zf,
                data.mse.mmse,
                data.mse.ai
            ],

            type: "bar",

            marker: {
                color: [
                    "#06b6d4",
                    "#8b5cf6",
                    "#ec4899"
                ]
            }
        }],

        {
            ...plotLayout,

            yaxis: {
                title: "MSE",
                gridcolor:
                    "rgba(148,163,184,.10)"
            }
        },

        {
            responsive: true,
            displayModeBar: false
        }
    );


    // Channel

    const subcarriers =
        Array.from(
            {length: 64},
            (_, i) => i
        );


    Plotly.newPlot(
        "channelMagnitude",

        [{
            x: subcarriers,

            y: data.channel.magnitude,

            type: "scatter",

            mode: "lines",

            line: {
                width: 3,
                color: "#00f5ff"
            },

            fill: "tozeroy",

            fillcolor:
                "rgba(0,245,255,.08)"
        }],

        {
            ...plotLayout,

            xaxis: {
                title: "Subcarrier"
            },

            yaxis: {
                title: "|H(f)|"
            }
        },

        {
            responsive: true,
            displayModeBar: false
        }
    );


    Plotly.newPlot(
        "channelPhase",

        [{
            x: subcarriers,

            y: data.channel.phase,

            type: "scatter",

            mode: "lines",

            line: {
                width: 3,
                color: "#c084fc"
            }
        }],

        {
            ...plotLayout,

            xaxis: {
                title: "Subcarrier"
            },

            yaxis: {
                title: "Phase (rad)"
            }
        },

        {
            responsive: true,
            displayModeBar: false
        }
    );


    // Constellations

    drawConstellation(
        "txConstellation",
        data.constellation.tx,
        "Transmitted"
    );

    drawConstellation(
        "rxConstellation",
        data.constellation.rx,
        "Received"
    );

    drawConstellation(
        "aiConstellation",
        data.constellation.ai,
        "AI Recovered"
    );

}


// ============================================================
// CONSTELLATION
// ============================================================

function drawConstellation(
    element,
    points,
    name
) {

    const x =
        points.map(
            p => p.x
        );

    const y =
        points.map(
            p => p.y
        );

    Plotly.newPlot(
        element,

        [{
            x: x,
            y: y,

            type: "scatter",

            mode: "markers",

            name: name,

            marker: {
                size: 9,
                opacity: .85,
                color: "#22d3ee"
            }
        }],

        {
            ...plotLayout,

            xaxis: {
                title: "In-Phase",
                zeroline: true
            },

            yaxis: {
                title: "Quadrature",
                zeroline: true,

                scaleanchor: "x"
            }
        },

        {
            responsive: true,
            displayModeBar: false
        }
    );
}


// ============================================================
// AI NEURAL NETWORK ANIMATION
// ============================================================

function drawNeural() {

    const canvas =
        document.getElementById(
            "neuralCanvas"
        );

    if (!canvas) {
        return;
    }

    const rect =
        canvas.getBoundingClientRect();

    canvas.width =
        rect.width * 2;

    canvas.height =
        rect.height * 2;

    const ctx =
        canvas.getContext("2d");

    ctx.scale(2, 2);

    const width =
        rect.width;

    const height =
        rect.height;

    const layers = [
        5,
        7,
        7,
        4
    ];

    const nodes = [];

    for (
        let layer = 0;
        layer < layers.length;
        layer++
    ) {

        const layerNodes = [];

        const count =
            layers[layer];

        const x =
            60 +
            layer *
            (
                (width - 120) /
                (layers.length - 1)
            );

        for (
            let i = 0;
            i < count;
            i++
        ) {

            const y =
                height / 2 +
                (
                    i -
                    (count - 1) / 2
                ) *
                38;

            layerNodes.push({
                x: x,
                y: y
            });
        }

        nodes.push(layerNodes);
    }

    let pulse = 0;

    function animate() {

        ctx.clearRect(
            0,
            0,
            width,
            height
        );

        // connections

        for (
            let l = 0;
            l < nodes.length - 1;
            l++
        ) {

            for (
                const a of nodes[l]
            ) {

                for (
                    const b of nodes[l + 1]
                ) {

                    ctx.beginPath();

                    ctx.moveTo(
                        a.x,
                        a.y
                    );

                    ctx.lineTo(
                        b.x,
                        b.y
                    );

                    ctx.strokeStyle =
                        "rgba(0,245,255,.08)";

                    ctx.lineWidth = 1;

                    ctx.stroke();
                }
            }
        }


        // pulse wave

        pulse += .035;

        // nodes

        for (
            let l = 0;
            l < nodes.length;
            l++
        ) {

            for (
                let i = 0;
                i < nodes[l].length;
                i++
            ) {

                const n =
                    nodes[l][i];

                const glow =
                    7 +
                    Math.sin(
                        pulse * 2 +
                        l +
                        i
                    ) * 4;

                ctx.beginPath();

                ctx.arc(
                    n.x,
                    n.y,
                    7,
                    0,
                    Math.PI * 2
                );

                ctx.shadowBlur =
                    glow * 2;

                ctx.shadowColor =
                    "#00f5ff";

                ctx.fillStyle =
                    "#22d3ee";

                ctx.fill();

                ctx.shadowBlur = 0;
            }
        }


        requestAnimationFrame(
            animate
        );
    }

    animate();
}


// ============================================================
// INITIAL DEMO SIMULATION
// ============================================================

setTimeout(
    () => {

        runSimulation();

    },
    1000
);

</script>

</body>

</html>
"""


# ============================================================
# FLASK ROUTES
# ============================================================

@app.route("/")
def index():

    if not session.get("logged_in"):
        return redirect(
            url_for("login")
        )

    return render_template_string(
        DASHBOARD_HTML
    )


@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        ).strip()

        if (
            username == USERNAME
            and
            password == PASSWORD
        ):

            session["logged_in"] = True

            return redirect(
                url_for("index")
            )

        return render_template_string(
            LOGIN_HTML,
            error="Invalid username or password"
        )

    return render_template_string(
        LOGIN_HTML,
        error=None
    )


@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("login")
    )


@app.route("/simulate")
def simulate():

    if not session.get("logged_in"):

        return jsonify({
            "error": "Unauthorized"
        }), 401

    try:

        snr = float(
            request.args.get(
                "snr",
                10
            )
        )

        frames = int(
            request.args.get(
                "frames",
                30
            )
        )

        data = run_simulation(
            selected_snr=snr,
            frames=frames
        )

        return jsonify(data)

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


@app.route("/health")
def health():

    return jsonify({
        "status": "online",
        "project": "NEXUS-WAVE AI",
        "model": "AI-MLP",
        "subcarriers": N_SUBCARRIERS
    })


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("                 NEXUS-WAVE AI")
    print("     Intelligent 3D OFDM Communication Laboratory")
    print("=" * 70)
    print()
    print("LOGIN DETAILS")
    print("-----------------------------")
    print("Username : admin")
    print("Password : ofdm2026")
    print()
    print("Preparing AI model...")
    print()

    train_ai_model()

    URL = "http://127.0.0.1:5000"

    def open_browser():

        try:

            webbrowser.open_new(
                URL
            )

        except Exception as e:

            print(
                "Browser could not be opened automatically:"
            )

            print(e)


    print()
    print("=" * 70)
    print("SERVER READY")
    print("=" * 70)
    print()
    print("Opening browser...")
    print("URL:", URL)
    print()
    print("Keep this terminal open while using the dashboard.")
    print("Press CTRL+C to stop the server.")
    print()

    threading.Timer(
        1.5,
        open_browser
    ).start()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )