# Real‑Time Human Activity Recognition (HAR) System

An end‑to‑end pipeline for real‑time human activity recognition using smartphone inertial sensors, built around a Temporal Convolutional Network (TCN). This project includes:

- **Sensor Logger App** (Android) – custom data collection tool  
- **Har Classifier App** (Android) – real‑time inference & display  
- **Prediction Server** (Python/Flask) – WebSocket backend running the trained TCN  

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

## 📱 Har Classifier App Usage

1. Install **Har Classifier** on your Android device.  
2. Enter the **Server IP** and **Port** (e.g., `192.168.1.100:5000`).  
3. Tap **Connect** to start streaming sensor data.  
4. View live **“Predicted Activity: ___”** updates on the screen.  

## 📂 Drive Links

- **PPT Presentation**: [Download the slides](https://docs.google.com/presentation/d/12hkaRYKkBgDTHmYy-fuqzSruyHd_J85eP36iDvF691g/edit?slide=id.g34f38648671_0_38#slide=id.g34f38648671_0_38)  
- **Project Report (PDF)**: [Download the report](https://drive.google.com/file/d/1qsoqbzCUXe_BxZ8T2tX2JuF5tg-JAVzi/view?usp=sharing)  
