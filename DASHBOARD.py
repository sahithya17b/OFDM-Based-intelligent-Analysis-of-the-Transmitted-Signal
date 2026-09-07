from flask import Flask, request, jsonify, render_template_string
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import threading
import webbrowser
import time

# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

# ============================================================
# PROJECT SETTINGS
# ============================================================

N_SUBCARRIERS = 64
CP_LEN = 16

NUM_TRAIN_FRAMES = 300
NUM_TEST_FRAMES = 50

SNR_TRAIN = np.array([0, 5, 10, 15, 20])
SNR_TEST = np.array([-5, 0, 5, 10, 15, 20, 25])

np.random.seed(42)


# ============================================================
# QPSK MODULATION
# ============================================================

def qpsk_mod(bits):

    bits = np.asarray(bits).reshape(-1, 2)

    symbols = np.zeros(len(bits), dtype=complex)

    for i, b in enumerate(bits):

        b1 = int(b[0])
        b2 = int(b[1])

        if b1 == 0 and b2 == 0:
            symbols[i] = (1 + 1j) / np.sqrt(2)

        elif b1 == 0 and b2 == 1:
            symbols[i] = (1 - 1j) / np.sqrt(2)

        elif b1 == 1 and b2 == 1:
            symbols[i] = (-1 - 1j) / np.sqrt(2)

        else:
            symbols[i] = (-1 + 1j) / np.sqrt(2)

    return symbols


# ============================================================
# QPSK DEMODULATION
# ============================================================

def qpsk_demod(symbols):

    bits = []

    for s in symbols:

        if s.real >= 0 and s.imag >= 0:
            bits.extend([0, 0])

        elif s.real >= 0 and s.imag < 0:
            bits.extend([0, 1])

        elif s.real < 0 and s.imag < 0:
            bits.extend([1, 1])

        else:
            bits.extend([1, 0])

    return np.array(bits, dtype=int)


# ============================================================
# OFDM TRANSMITTER
# ============================================================

def ofdm_transmitter():

    bits = np.random.randint(
        0,
        2,
        2 * N_SUBCARRIERS
    )

    symbols = qpsk_mod(bits)

    time_signal = np.fft.ifft(symbols)

    cyclic_prefix = time_signal[-CP_LEN:]

    tx_signal = np.concatenate(
        (cyclic_prefix, time_signal)
    )

    return bits, symbols, tx_signal


# ============================================================
# CHANNEL
# ============================================================

def channel(tx_signal, snr_db):

    path_gains = np.array([
        0.9,
        0.4,
        0.2
    ])

    h = (
        np.random.randn(len(path_gains))
        +
        1j * np.random.randn(len(path_gains))
    ) / np.sqrt(2)

    h = h * path_gains

    h = h / np.sqrt(
        np.sum(np.abs(h) ** 2)
    )

    faded_signal = np.convolve(
        tx_signal,
        h
    )

    faded_signal = faded_signal[
        :len(tx_signal)
    ]

    signal_power = np.mean(
        np.abs(faded_signal) ** 2
    )

    snr_linear = 10 ** (snr_db / 10)

    noise_power = (
        signal_power / snr_linear
    )

    noise = np.sqrt(
        noise_power / 2
    ) * (
        np.random.randn(len(faded_signal))
        +
        1j * np.random.randn(len(faded_signal))
    )

    rx_signal = faded_signal + noise

    return rx_signal, h


# ============================================================
# OFDM RECEIVER
# ============================================================

def ofdm_receiver(rx_signal):

    useful_signal = rx_signal[
        CP_LEN:
        CP_LEN + N_SUBCARRIERS
    ]

    received_symbols = np.fft.fft(
        useful_signal
    )

    return received_symbols


# ============================================================
# CHANNEL FREQUENCY RESPONSE
# ============================================================

def channel_frequency_response(h):

    h_padded = np.zeros(
        N_SUBCARRIERS,
        dtype=complex
    )

    h_padded[:len(h)] = h

    H = np.fft.fft(h_padded)

    return H


# ============================================================
# ZERO FORCING
# ============================================================

def zf_equalizer(Y, H):

    epsilon = 1e-8

    return Y / (H + epsilon)


# ============================================================
# MMSE
# ============================================================

def mmse_equalizer(Y, H, snr_db):

    snr_linear = 10 ** (
        snr_db / 10
    )

    noise_variance = 1 / snr_linear

    W = (
        np.conj(H)
        /
        (
            np.abs(H) ** 2
            +
            noise_variance
        )
    )

    return W * Y


# ============================================================
# GENERATE SAMPLE
# ============================================================

def generate_sample(snr_db):

    bits, tx_symbols, tx_signal = (
        ofdm_transmitter()
    )

    rx_signal, h = channel(
        tx_signal,
        snr_db
    )

    Y = ofdm_receiver(
        rx_signal
    )

    H = channel_frequency_response(
        h
    )

    ZF = zf_equalizer(
        Y,
        H
    )

    MMSE = mmse_equalizer(
        Y,
        H,
        snr_db
    )

    return (
        bits,
        tx_symbols,
        Y,
        H,
        ZF,
        MMSE
    )


# ============================================================
# AI TRAINING
# ============================================================

print()
print("==============================================")
print("     AI-BASED OFDM CHANNEL EQUALIZER")
print("==============================================")
print()
print("Preparing AI training data...")
print()

X_train = []
y_train = []

for frame in range(NUM_TRAIN_FRAMES):

    snr = np.random.choice(
        SNR_TRAIN
    )

    (
        bits,
        tx_symbols,
        Y,
        H,
        ZF,
        MMSE
    ) = generate_sample(snr)

    for k in range(N_SUBCARRIERS):

        X_train.append([
            Y[k].real,
            Y[k].imag,
            H[k].real,
            H[k].imag,
            snr / 20.0
        ])

        y_train.append([
            tx_symbols[k].real,
            tx_symbols[k].imag
        ])


X_train = np.array(X_train)
y_train = np.array(y_train)

print("Training samples:", len(X_train))
print()
print("Scaling training data...")

input_scaler = StandardScaler()
output_scaler = StandardScaler()

X_train_scaled = input_scaler.fit_transform(
    X_train
)

y_train_scaled = output_scaler.fit_transform(
    y_train
)

print("Training neural network...")
print()

model = MLPRegressor(

    hidden_layer_sizes=(64, 64),

    activation="relu",

    solver="adam",

    learning_rate_init=0.001,

    batch_size=512,

    max_iter=100,

    random_state=42,

    verbose=False
)

model.fit(
    X_train_scaled,
    y_train_scaled
)

print()
print("==============================================")
print("       AI TRAINING COMPLETED")
print("==============================================")
print()


# ============================================================
# AI EQUALIZER
# ============================================================

def ai_equalizer(Y, H, snr_db):

    X = []

    for k in range(N_SUBCARRIERS):

        X.append([
            Y[k].real,
            Y[k].imag,
            H[k].real,
            H[k].imag,
            snr_db / 20.0
        ])

    X = np.array(X)

    X_scaled = input_scaler.transform(
        X
    )

    prediction_scaled = model.predict(
        X_scaled
    )

    prediction = output_scaler.inverse_transform(
        prediction_scaled
    )

    return (
        prediction[:, 0]
        +
        1j * prediction[:, 1]
    )


# ============================================================
# BER
# ============================================================

def calculate_ber(
    original_bits,
    estimated_symbols
):

    estimated_bits = qpsk_demod(
        estimated_symbols
    )

    return float(
        np.mean(
            original_bits != estimated_bits
        )
    )


# ============================================================
# MSE
# ============================================================

def calculate_mse(
    original_symbols,
    estimated_symbols
):

    return float(
        np.mean(
            np.abs(
                original_symbols -
                estimated_symbols
            ) ** 2
        )
    )


# ============================================================
# RUN SIMULATION
# ============================================================

def run_simulation(selected_snr):

    results = {

        "ZF": {
            "BER": [],
            "MSE": []
        },

        "MMSE": {
            "BER": [],
            "MSE": []
        },

        "AI": {
            "BER": [],
            "MSE": []
        }
    }

    print()
    print(
        f"Running simulation for selected SNR = "
        f"{selected_snr} dB"
    )

    # ========================================================
    # SNR LOOP
    # ========================================================

    for snr in SNR_TEST:

        print(
            f"  Processing SNR = {snr} dB..."
        )

        ber_zf = []
        ber_mmse = []
        ber_ai = []

        mse_zf = []
        mse_mmse = []
        mse_ai = []

        for frame in range(NUM_TEST_FRAMES):

            (
                bits,
                tx_symbols,
                Y,
                H,
                ZF,
                MMSE
            ) = generate_sample(snr)

            AI = ai_equalizer(
                Y,
                H,
                snr
            )

            ber_zf.append(
                calculate_ber(
                    bits,
                    ZF
                )
            )

            ber_mmse.append(
                calculate_ber(
                    bits,
                    MMSE
                )
            )

            ber_ai.append(
                calculate_ber(
                    bits,
                    AI
                )
            )

            mse_zf.append(
                calculate_mse(
                    tx_symbols,
                    ZF
                )
            )

            mse_mmse.append(
                calculate_mse(
                    tx_symbols,
                    MMSE
                )
            )

            mse_ai.append(
                calculate_mse(
                    tx_symbols,
                    AI
                )
            )

        results["ZF"]["BER"].append(
            float(np.mean(ber_zf))
        )

        results["MMSE"]["BER"].append(
            float(np.mean(ber_mmse))
        )

        results["AI"]["BER"].append(
            float(np.mean(ber_ai))
        )

        results["ZF"]["MSE"].append(
            float(np.mean(mse_zf))
        )

        results["MMSE"]["MSE"].append(
            float(np.mean(mse_mmse))
        )

        results["AI"]["MSE"].append(
            float(np.mean(mse_ai))
        )

    # ========================================================
    # CONSTELLATION
    # ========================================================

    (
        bits,
        tx_symbols,
        Y,
        H,
        ZF,
        MMSE
    ) = generate_sample(
        selected_snr
    )

    AI = ai_equalizer(
        Y,
        H,
        selected_snr
    )

    # ========================================================
    # SELECTED SNR
    # ========================================================

    selected = {

        "snr":
            selected_snr,

        "ber_zf":
            calculate_ber(
                bits,
                ZF
            ),

        "ber_mmse":
            calculate_ber(
                bits,
                MMSE
            ),

        "ber_ai":
            calculate_ber(
                bits,
                AI
            ),

        "mse_zf":
            calculate_mse(
                tx_symbols,
                ZF
            ),

        "mse_mmse":
            calculate_mse(
                tx_symbols,
                MMSE
            ),

        "mse_ai":
            calculate_mse(
                tx_symbols,
                AI
            )
    }

    # ========================================================
    # COMPLEX → X/Y
    # ========================================================

    def complex_to_xy(symbols):

        return {

            "x":
                symbols.real.tolist(),

            "y":
                symbols.imag.tolist()
        }

    constellation = {

        "Transmitted":
            complex_to_xy(
                tx_symbols
            ),

        "Received":
            complex_to_xy(
                Y
            ),

        "ZF":
            complex_to_xy(
                ZF
            ),

        "MMSE":
            complex_to_xy(
                MMSE
            ),

        "AI":
            complex_to_xy(
                AI
            )
    }

    print("Simulation completed.")

    return {

        "snr":
            SNR_TEST.tolist(),

        "ber": {

            "ZF":
                results["ZF"]["BER"],

            "MMSE":
                results["MMSE"]["BER"],

            "AI":
                results["AI"]["BER"]
        },

        "mse": {

            "ZF":
                results["ZF"]["MSE"],

            "MMSE":
                results["MMSE"]["MSE"],

            "AI":
                results["AI"]["MSE"]
        },

        "selected":
            selected,

        "constellation":
            constellation
    }


# ============================================================
# HTML DASHBOARD
# ============================================================

HTML = r"""

<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<meta name="viewport"
content="width=device-width, initial-scale=1.0">

<title>
AI OFDM Communication Dashboard
</title>

<script src="https://cdn.plot.ly/plotly-2.35.2.min.js">
</script>

<style>

/* ============================================================
   GLOBAL
============================================================ */

* {
    box-sizing: border-box;
}

body {

    margin: 0;

    font-family:
    Arial,
    Helvetica,
    sans-serif;

    background:
    linear-gradient(
        135deg,
        #eef2ff,
        #f0fdfa,
        #fff7ed
    );

    color: #172033;
}


/* ============================================================
   HEADER
============================================================ */

.header {

    position: relative;

    overflow: hidden;

    background:
    linear-gradient(
        135deg,
        #111827,
        #312e81,
        #4f46e5,
        #7c3aed,
        #db2777
    );

    color: white;

    padding:
    45px 25px;

    text-align: center;

    box-shadow:
    0 10px 35px
    rgba(79,70,229,0.35);
}

.header h1 {

    margin: 0 0 12px;

    font-size: 38px;

    position: relative;

    z-index: 2;
}

.header p {

    margin: 0;

    font-size: 17px;

    opacity: 0.9;

    position: relative;

    z-index: 2;
}


/* ============================================================
   CONTAINER
============================================================ */

.container {

    max-width: 1450px;

    margin: auto;

    padding: 30px;
}


/* ============================================================
   INFO
============================================================ */

.info {

    background:
    rgba(255,255,255,0.95);

    padding: 28px;

    border-radius: 20px;

    box-shadow:
    0 10px 35px
    rgba(31,41,55,0.10);

    margin-bottom: 25px;

    border-left:
    6px solid #6366f1;
}

.info h2 {

    margin-top: 0;

    color: #4338ca;
}


/* ============================================================
   CONTROL
============================================================ */

.control-panel {

    background:
    linear-gradient(
        135deg,
        #ffffff,
        #eef2ff
    );

    padding: 25px;

    border-radius: 20px;

    box-shadow:
    0 10px 30px
    rgba(31,41,55,0.10);

    margin-bottom: 25px;

    display: flex;

    align-items: center;

    gap: 15px;

    flex-wrap: wrap;
}

.control-panel label {

    font-weight: bold;

    color: #312e81;
}

select {

    padding:
    12px 18px;

    border:
    2px solid #c7d2fe;

    border-radius: 10px;

    font-size: 16px;

    font-weight: bold;

    color: #312e81;

    background: white;
}

button {

    padding:
    13px 28px;

    border: none;

    border-radius: 10px;

    background:
    linear-gradient(
        135deg,
        #4f46e5,
        #7c3aed
    );

    color: white;

    font-size: 15px;

    font-weight: bold;

    cursor: pointer;

    box-shadow:
    0 6px 15px
    rgba(79,70,229,0.3);

    transition: 0.25s;
}

button:hover {

    transform:
    translateY(-3px);

    box-shadow:
    0 12px 25px
    rgba(79,70,229,0.4);
}

.loading {

    display: none;

    color: #4f46e5;

    font-weight: bold;
}


/* ============================================================
   CARDS
============================================================ */

.cards {

    display: grid;

    grid-template-columns:
    repeat(3, 1fr);

    gap: 22px;

    margin-bottom: 25px;
}

.card {

    background: white;

    padding: 25px;

    border-radius: 20px;

    box-shadow:
    0 10px 30px
    rgba(31,41,55,0.10);

    transition: 0.25s;
}

.card:hover {

    transform:
    translateY(-5px);

    box-shadow:
    0 15px 35px
    rgba(31,41,55,0.16);
}

.zf-card {

    border-top:
    7px solid #ef4444;
}

.mmse-card {

    border-top:
    7px solid #10b981;
}

.ai-card {

    border-top:
    7px solid #8b5cf6;
}

.zf-card h3 {
    color: #dc2626;
}

.mmse-card h3 {
    color: #059669;
}

.ai-card h3 {
    color: #7c3aed;
}

.value {

    font-size: 28px;

    font-weight: bold;

    margin: 8px 0;

    color: #111827;
}

.small {

    color: #6b7280;

    font-size: 14px;

    margin-bottom: 4px;
}


/* ============================================================
   CHARTS
============================================================ */

.chart-card {

    background:
    rgba(255,255,255,0.96);

    padding: 25px;

    border-radius: 20px;

    box-shadow:
    0 10px 35px
    rgba(31,41,55,0.10);

    margin-bottom: 25px;
}

.chart-card h2 {

    margin-top: 0;

    color: #312e81;
}

.chart {

    width: 100%;

    height: 520px;
}


/* ============================================================
   CONSTELLATION
============================================================ */

.constellation-grid {

    display: grid;

    grid-template-columns:
    repeat(3, 1fr);

    gap: 22px;
}

.constellation-card {

    background:
    linear-gradient(
        135deg,
        #ffffff,
        #f8fafc
    );

    border:
    2px solid #e5e7eb;

    padding: 15px;

    border-radius: 18px;
}

.constellation {

    width: 100%;

    height: 380px;
}


/* ============================================================
   ARCHITECTURE
============================================================ */

.architecture {

    background:
    linear-gradient(
        135deg,
        #111827,
        #312e81,
        #4f46e5,
        #7c3aed
    );

    color: white;

    padding: 30px;

    border-radius: 20px;

    box-shadow:
    0 10px 35px
    rgba(79,70,229,0.30);

    margin-bottom: 25px;
}

.architecture h2 {

    color: white;
}

.flow {

    padding: 22px;

    background:
    rgba(255,255,255,0.12);

    border-radius: 15px;

    font-size: 17px;

    line-height: 2;

    text-align: center;
}


/* ============================================================
   FOOTER
============================================================ */

.footer {

    text-align: center;

    padding: 30px;

    color: white;

    background:
    linear-gradient(
        135deg,
        #111827,
        #312e81
    );
}


/* ============================================================
   RESPONSIVE
============================================================ */

@media(max-width:1000px) {

    .cards {

        grid-template-columns: 1fr;
    }

    .constellation-grid {

        grid-template-columns: 1fr;
    }

    .header h1 {

        font-size: 28px;
    }

}

</style>

</head>


<body>


<!-- ============================================================
     HEADER
============================================================ -->

<div class="header">

<h1>
🤖 AI-Based Adaptive Channel Equalization
</h1>

<p>
📡 OFDM Communication System under Fading and Noisy Channels
</p>

</div>


<div class="container">


<!-- ============================================================
     PROJECT OVERVIEW
============================================================ -->

<div class="info">

<h2>
📡 Project Overview
</h2>

<p>

This interactive dashboard demonstrates an
<strong>AI-based OFDM channel equalization system</strong>
operating under multipath fading and AWGN noise.

</p>

<p>
The system compares three equalization techniques:
</p>

<ul>

<li>
🔴 <strong>Zero Forcing (ZF)</strong>
</li>

<li>
🟢 <strong>Minimum Mean Square Error (MMSE)</strong>
</li>

<li>
🟣 <strong>Artificial Intelligence Equalizer</strong>
</li>

</ul>

<p>

Performance is evaluated using:

<strong>Bit Error Rate (BER)</strong>

and

<strong>Mean Square Error (MSE)</strong>.

</p>

</div>


<!-- ============================================================
     CONTROLS
============================================================ -->

<div class="control-panel">

<label>
🎛️ Select SNR:
</label>

<select id="snr">

<option value="-5">
-5 dB
</option>

<option value="0">
0 dB
</option>

<option value="5">
5 dB
</option>

<option value="10" selected>
10 dB
</option>

<option value="15">
15 dB
</option>

<option value="20">
20 dB
</option>

<option value="25">
25 dB
</option>

</select>

<button onclick="runSimulation()">

🚀 RUN SIMULATION

</button>

<div
class="loading"
id="loading">

⏳ Running OFDM simulation...
Please wait...

</div>

</div>


<!-- ============================================================
     KPI CARDS
============================================================ -->

<div class="cards">


<div class="card zf-card">

<h3>
🔴 ZF Equalizer
</h3>

<div class="small">
Bit Error Rate
</div>

<div
class="value"
id="zfBer">
--
</div>

<div class="small">
Mean Square Error
</div>

<div
class="value"
id="zfMse">
--
</div>

</div>


<div class="card mmse-card">

<h3>
🟢 MMSE Equalizer
</h3>

<div class="small">
Bit Error Rate
</div>

<div
class="value"
id="mmseBer">
--
</div>

<div class="small">
Mean Square Error
</div>

<div
class="value"
id="mmseMse">
--
</div>

</div>


<div class="card ai-card">

<h3>
🟣 AI Equalizer
</h3>

<div class="small">
Bit Error Rate
</div>

<div
class="value"
id="aiBer">
--
</div>

<div class="small">
Mean Square Error
</div>

<div
class="value"
id="aiMse">
--
</div>

</div>


</div>


<!-- ============================================================
     BER CHART
============================================================ -->

<div class="chart-card">

<h2>
📉 Bit Error Rate vs SNR
</h2>

<div
id="berChart"
class="chart">
</div>

</div>


<!-- ============================================================
     MSE CHART
============================================================ -->

<div class="chart-card">

<h2>
📊 Mean Square Error vs SNR
</h2>

<div
id="mseChart"
class="chart">
</div>

</div>


<!-- ============================================================
     CONSTELLATION
============================================================ -->

<div class="chart-card">

<h2>
✨ Constellation Diagrams
</h2>

<div class="constellation-grid">


<div class="constellation-card">

<h3>
📤 Transmitted
</h3>

<div
id="transmitted"
class="constellation">
</div>

</div>


<div class="constellation-card">

<h3>
📥 Received
</h3>

<div
id="received"
class="constellation">
</div>

</div>


<div class="constellation-card">

<h3>
🔴 ZF Equalized
</h3>

<div
id="zfConstellation"
class="constellation">
</div>

</div>


<div class="constellation-card">

<h3>
🟢 MMSE Equalized
</h3>

<div
id="mmseConstellation"
class="constellation">
</div>

</div>


<div class="constellation-card">

<h3>
🟣 AI Equalized
</h3>

<div
id="aiConstellation"
class="constellation">
</div>

</div>


</div>

</div>


<!-- ============================================================
     ARCHITECTURE
============================================================ -->

<div class="architecture">

<h2>
⚙️ OFDM System Architecture
</h2>

<div class="flow">

🎲 Random Bits
→
🔵 QPSK
→
🔄 IFFT
→
➕ Cyclic Prefix
→
📡 Fading Channel
→
🌊 AWGN
→
🔄 FFT
→
⚙️ Equalization
→
🔵 QPSK Demodulation
→
📈 BER

</div>

<br>

<h2>
🧠 Equalizers Compared
</h2>

<ul>

<li>

<strong>ZF:</strong>
Directly compensates for channel distortion.

</li>

<li>

<strong>MMSE:</strong>
Considers both channel response and noise.

</li>

<li>

<strong>AI Equalizer:</strong>
Uses a neural network to learn the relationship
between received I/Q samples, channel response and SNR.

</li>

</ul>

</div>


</div>


<!-- ============================================================
     FOOTER
============================================================ -->

<div class="footer">

📡 <strong>AI-Based OFDM Channel Equalization</strong>

<br><br>

ECE Major Project
&nbsp;|&nbsp;
Artificial Intelligence
&nbsp;|&nbsp;
Wireless Communication
&nbsp;|&nbsp;
OFDM

</div>


<script>


// ============================================================
// RUN SIMULATION
// ============================================================

function runSimulation() {

    const snr =
        document.getElementById(
            "snr"
        ).value;

    document.getElementById(
        "loading"
    ).style.display =
        "block";


    document.getElementById(
        "loading"
    ).innerText =
        "⏳ Running OFDM simulation... Please wait";


    fetch(
        "/simulate?snr=" + snr
    )

    .then(
        response => {

            if (!response.ok) {

                throw new Error(
                    "Server returned an error"
                );

            }

            return response.json();

        }
    )

    .then(
        data => {

            updateDashboard(
                data
            );

            document.getElementById(
                "loading"
            ).style.display =
                "none";

        }
    )

    .catch(
        error => {

            console.error(error);

            alert(
                "Simulation error. Check the Python terminal."
            );

            document.getElementById(
                "loading"
            ).style.display =
                "none";

        }
    );

}


// ============================================================
// UPDATE DASHBOARD
// ============================================================

function updateDashboard(data) {


    document.getElementById(
        "zfBer"
    ).innerText =
        data.selected.ber_zf.toFixed(6);


    document.getElementById(
        "mmseBer"
    ).innerText =
        data.selected.ber_mmse.toFixed(6);


    document.getElementById(
        "aiBer"
    ).innerText =
        data.selected.ber_ai.toFixed(6);


    document.getElementById(
        "zfMse"
    ).innerText =
        data.selected.mse_zf.toFixed(6);


    document.getElementById(
        "mmseMse"
    ).innerText =
        data.selected.mse_mmse.toFixed(6);


    document.getElementById(
        "aiMse"
    ).innerText =
        data.selected.mse_ai.toFixed(6);


    // ========================================================
    // BER GRAPH
    // ========================================================

    const berTraces = [

        {

            x: data.snr,

            y: data.ber.ZF,

            mode:
                "lines+markers",

            name:
                "ZF",

            line: {
                color:
                    "#ef4444",

                width:
                    4
            },

            marker: {
                size:
                    10
            }

        },


        {

            x: data.snr,

            y: data.ber.MMSE,

            mode:
                "lines+markers",

            name:
                "MMSE",

            line: {
                color:
                    "#10b981",

                width:
                    4
            },

            marker: {
                size:
                    10
            }

        },


        {

            x: data.snr,

            y: data.ber.AI,

            mode:
                "lines+markers",

            name:
                "AI Equalizer",

            line: {
                color:
                    "#8b5cf6",

                width:
                    4
            },

            marker: {
                size:
                    10
            }

        }

    ];


    Plotly.newPlot(

        "berChart",

        berTraces,

        {

            paper_bgcolor:
                "rgba(0,0,0,0)",

            plot_bgcolor:
                "#f8fafc",

            xaxis: {

                title:
                    "SNR (dB)",

                gridcolor:
                    "#e5e7eb"

            },

            yaxis: {

                title:
                    "Bit Error Rate",

                type:
                    "log",

                gridcolor:
                    "#e5e7eb"

            },

            hovermode:
                "x unified",

            legend: {

                orientation:
                    "h"

            },

            margin: {

                t:
                    30

            }

        },

        {

            responsive:
                true

        }

    );


    // ========================================================
    // MSE GRAPH
    // ========================================================

    const mseTraces = [

        {

            x:
                data.snr,

            y:
                data.mse.ZF,

            mode:
                "lines+markers",

            name:
                "ZF",

            line: {

                color:
                    "#ef4444",

                width:
                    4

            },

            marker: {

                size:
                    10

            }

        },


        {

            x:
                data.snr,

            y:
                data.mse.MMSE,

            mode:
                "lines+markers",

            name:
                "MMSE",

            line: {

                color:
                    "#10b981",

                width:
                    4

            },

            marker: {

                size:
                    10

            }

        },


        {

            x:
                data.snr,

            y:
                data.mse.AI,

            mode:
                "lines+markers",

            name:
                "AI Equalizer",

            line: {

                color:
                    "#8b5cf6",

                width:
                    4

            },

            marker: {

                size:
                    10

            }

        }

    ];


    Plotly.newPlot(

        "mseChart",

        mseTraces,

        {

            paper_bgcolor:
                "rgba(0,0,0,0)",

            plot_bgcolor:
                "#f8fafc",

            xaxis: {

                title:
                    "SNR (dB)",

                gridcolor:
                    "#e5e7eb"

            },

            yaxis: {

                title:
                    "Mean Square Error",

                type:
                    "log",

                gridcolor:
                    "#e5e7eb"

            },

            hovermode:
                "x unified",

            legend: {

                orientation:
                    "h"

            },

            margin: {

                t:
                    30

            }

        },

        {

            responsive:
                true

        }

    );


    // ========================================================
    // CONSTELLATIONS
    // ========================================================

    createConstellation(
        "transmitted",
        data.constellation.Transmitted,
        "Transmitted",
        "#2563eb"
    );


    createConstellation(
        "received",
        data.constellation.Received,
        "Received",
        "#f97316"
    );


    createConstellation(
        "zfConstellation",
        data.constellation.ZF,
        "ZF Equalized",
        "#ef4444"
    );


    createConstellation(
        "mmseConstellation",
        data.constellation.MMSE,
        "MMSE Equalized",
        "#10b981"
    );


    createConstellation(
        "aiConstellation",
        data.constellation.AI,
        "AI Equalized",
        "#8b5cf6"
    );

}


// ============================================================
// CONSTELLATION
// ============================================================

function createConstellation(
    element,
    values,
    title,
    pointColor
) {

    const trace = {

        x:
            values.x,

        y:
            values.y,

        mode:
            "markers",

        type:
            "scatter",

        marker: {

            size:
                10,

            color:
                pointColor,

            line: {

                width:
                    1,

                color:
                    "#111827"

            }

        }

    };


    Plotly.newPlot(

        element,

        [trace],

        {

            title: {

                text:
                    title,

                font: {

                    size:
                        18

                }

            },

            paper_bgcolor:
                "rgba(0,0,0,0)",

            plot_bgcolor:
                "#f8fafc",

            xaxis: {

                title:
                    "In-Phase (I)",

                zeroline:
                    true,

                gridcolor:
                    "#e5e7eb",

                scaleanchor:
                    "y"

            },

            yaxis: {

                title:
                    "Quadrature (Q)",

                zeroline:
                    true,

                gridcolor:
                    "#e5e7eb"

            },

            margin: {

                t:
                    55,

                l:
                    55,

                r:
                    20,

                b:
                    55

            },

            showlegend:
                false

        },

        {

            responsive:
                true

        }

    );

}


// ============================================================
// AUTOMATIC SIMULATION
// ============================================================

window.onload = function() {

    setTimeout(
        function() {

            runSimulation();

        },
        500
    );

};

</script>


</body>

</html>

"""


# ============================================================
# FLASK HOME
# ============================================================

@app.route("/")
def home():

    return render_template_string(
        HTML
    )


# ============================================================
# SIMULATION API
# ============================================================

@app.route("/simulate")
def simulate():

    try:

        snr = int(
            request.args.get(
                "snr",
                10
            )
        )

        if snr not in SNR_TEST:

            snr = 10

        results = run_simulation(
            snr
        )

        return jsonify(
            results
        )

    except Exception as e:

        print()
        print("ERROR:")
        print(e)
        print()

        return jsonify({

            "error":
                str(e)

        }), 500


# ============================================================
# AUTOMATIC BROWSER
# ============================================================

def open_browser():

    try:

        webbrowser.open(
            "http://127.0.0.1:5000/",
            new=1
        )

        print(
            "Browser opened automatically."
        )

    except Exception as e:

        print(
            "Could not open browser automatically:"
        )

        print(e)


# ============================================================
# START FLASK
# ============================================================

if __name__ == "__main__":

    print()
    print("==============================================")
    print("        AI OFDM DASHBOARD SERVER")
    print("==============================================")
    print()
    print("Starting Flask server...")
    print()
    print("Dashboard URL:")
    print("http://127.0.0.1:5000/")
    print()

    # Open browser AFTER server has time to start
    threading.Timer(
        2.0,
        open_browser
    ).start()

    app.run(

        host="127.0.0.1",

        port=5000,

        debug=False,

        use_reloader=False

    )