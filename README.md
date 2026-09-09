 OFDM-Based Intelligent Analysis of the Transmitted Signal

📡 Project Overview
*OFDM-Based Intelligent Analysis of the Transmitted Signal is a communication-system project that simulates an **Orthogonal Frequency Division Multiplexing (OFDM)** communication system and intelligently analyzes the transmitted and received signals.

The project demonstrates the complete OFDM transmission process, including signal generation, modulation, IFFT/FFT processing, cyclic prefix insertion and removal, channel effects, noise addition, signal recovery, and performance analysis.

An interactive dashboard is provided to visualize important communication parameters and analyze the system performance.

 🎯 Objectives

* Design and simulate an OFDM communication system.
* Generate and analyze OFDM transmitted signals.
* Perform QPSK-based digital modulation.
* Implement IFFT and FFT operations.
* Add and remove the cyclic prefix.
* Simulate wireless channel effects and noise.
* Recover the transmitted signal at the receiver.
* Calculate communication performance parameters.
* Analyze **Bit Error Rate (BER)** and signal quality.
* Provide an interactive dashboard for visualization.
* Demonstrate the practical working of an OFDM communication system.

---

## 🧠 Key Features

* OFDM transmitter and receiver simulation
* QPSK modulation and demodulation
* IFFT/FFT processing
* Cyclic Prefix implementation
* AWGN channel simulation
* Signal transmission and recovery
* BER calculation
* SNR analysis
* Constellation diagram
* Transmitted vs. received signal visualization
* Interactive web-based dashboard
* Intelligent signal analysis using machine learning

---

## 🏗️ System Architecture

```text
                ┌─────────────────────┐
                │   Input Data        │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │   QPSK Modulation   │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │   OFDM Subcarriers  │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │        IFFT         │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │   Cyclic Prefix     │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Wireless Channel    │
                │ + AWGN Noise        │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Remove Cyclic Prefix│
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │        FFT          │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ QPSK Demodulation   │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Received Data       │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Intelligent Analysis│
                │ & Visualization     │
                └─────────────────────┘
```

---

## ⚙️ Technologies Used

| Technology   | Purpose                                    |
| ------------ | ------------------------------------------ |
| Python       | Core implementation                        |
| NumPy        | Numerical and signal processing operations |
| Scikit-learn | Machine learning / intelligent analysis    |
| Flask        | Web dashboard                              |
| HTML         | Dashboard structure                        |
| CSS          | Dashboard styling                          |
| JavaScript   | Interactive visualization                  |
| Matplotlib   | Signal and performance graphs              |

---

## 📊 OFDM Parameters

The simulation uses configurable OFDM parameters such as:

```text
Number of Subcarriers : 64
Cyclic Prefix Length  : 16
Modulation            : QPSK
Channel               : AWGN
SNR                   : Configurable
Training Frames       : 300
```

These parameters can be modified according to the required simulation conditions.

---

## 📈 Performance Metrics

The project analyzes several important communication parameters.

### Bit Error Rate (BER)

BER measures the number of incorrectly received bits compared with the total transmitted bits.

```text
BER = Number of Bit Errors / Total Number of Transmitted Bits
```

A lower BER indicates better communication performance.

### Signal-to-Noise Ratio (SNR)

SNR represents the ratio between signal power and noise power.

Higher SNR generally results in better signal recovery and lower BER.

### Constellation Analysis

The QPSK constellation is used to visualize the effect of noise and channel impairments on the received symbols.

---

## 📉 Dashboard Visualizations

The dashboard provides visualization of:

* Transmitted signal
* Received signal
* QPSK constellation
* BER vs SNR
* Signal power
* Noise power
* OFDM subcarrier information
* Signal quality analysis
* System performance
  
 🚀 How to Run the Project

1. Clone the Repository
 2. Navigate to the Project
 3. Install Required Libraries
4. Run the Application
 5. Open the Dashboard
Open your browser and visit:text
http://127.0.0.1:5000


-
