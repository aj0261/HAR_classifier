# Real‑Time Human Activity Recognition (HAR) System

An end‑to‑end pipeline for real‑time human activity recognition using smartphone inertial sensors, built around a Temporal Convolutional Network (TCN). This project includes:

- **Sensor Logger App** (Android) – custom data collection tool  
- **Har Classifier App** (Android) – real‑time inference & display  
- **Prediction Server** (Python/Flask) – WebSocket backend running the trained TCN  

---

## 📖 Table of Contents

1. [Features](#features)  
2. [Architecture](#architecture)  
3. [Prerequisites](#prerequisites)  
4. [Installation](#installation)  
5. [Data Collection](#data-collection)  
6. [Model Training](#model-training)  
7. [Running the Prediction Server](#running-the-prediction-server)  
8. [Har Classifier App Usage](#har-classifier-app-usage)  
9. [Drive Links](#drive-links)  
10. [Future Scope](#future-scope)  
11. [Authors & License](#authors--license)  

---

## ✨ Features

- **Real‑Time Inference**: <50 ms per 3 s window  
- **Custom Data Collection**: Android app samples accelerometer & gyroscope at 20 Hz  
- **WebSocket Communication**: low‑latency, bi‑directional streaming  
- **Extensible Architecture**: easily add new activities by collecting more data and retraining  

---

## 🏗 Architecture

```text
┌───────────────────┐     ┌───────────┐     ┌──────────────────┐     ┌───────────┐
│ Sensor Logger     │     │ Prediction│     │ ML Processing    │     │ Dashboard │
│ Android App       │<--->│ Server    │<--->│ (TCN Inference)  │<--->│ (Optional)│
└───────────────────┘     └───────────┘     └──────────────────┘     └───────────┘
