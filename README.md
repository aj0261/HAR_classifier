# Real‑Time Human Activity Recognition (HAR) System

An end‑to‑end pipeline for real‑time human activity recognition using smartphone inertial sensors, built around a Temporal Convolutional Network (TCN). This project includes:

- **Sensor Logger App** (Android) – custom data collection tool  
- **Har Classifier App** (Android) – real‑time inference & display  
- **Prediction Server** (Python/Flask) – WebSocket backend running the trained TCN  

---

## ✨ Features

- **Real‑Time Inference**: <50 ms per 3 s window  
- **Custom Data Collection**: Android app samples accelerometer & gyroscope at 20 Hz  
- **WebSocket Communication**: low‑latency, bi‑directional streaming  
- **Extensible Architecture**: easily add new activities by collecting more data and retraining  

---

### Technology Stack
- **Frontend**: Java/Kotlin, Android SDK
- **Backend**: Python, Flask
- **ML Framework**: TensorFlow/PyTorch

---

## 🏗 Architecture

```text
┌───────────────────┐     ┌───────────┐     ┌──────────────────┐     ┌───────────┐
│ Sensor Logger     │     │ Prediction│     │ ML Processing    │     │ Dashboard │
│ Android App       │<--->│ Server    │<--->│ (TCN Inference)  │<--->│ (Optional)│
└───────────────────┘     └───────────┘     └──────────────────┘     └───────────┘
```

---

## 📱 Har Classifier App Usage
1. Install **Har Classifier** on your Android device.  
2. Enter the **Server IP** and **Port** (e.g., `192.168.1.100:5000`).  
3. Tap **Connect** to start streaming sensor data.  
4. View live **"Predicted Activity: ___"** updates on the screen.  

---

## 🔧 System Requirements
- Android 7.0 or higher
- Internet connection
- Access to device sensors (accelerometer, gyroscope)

---

## 📈 Supported Activities
- Walking
- Running
- Sitting
- Standing
- Going upstairs
- Going downstairs

---



## ⚙️ Getting Started

### Prerequisites
- Android Studio 4.0+
- JDK 8+
- Python 3.7+ (for server)

### Installation
1. Clone this repository:
```bash
git clone https://github.com/yourusername/har-classifier.git
cd har-classifier
```

2. Open the project in Android Studio and build the app.

3. Set up the server:
```bash
cd server
pip install -r requirements.txt
python app.py
```

---
## 📂 Project Resources

### Drive Links
- **PPT Presentation**: [Download the slides](https://docs.google.com/presentation/d/12hkaRYKkBgDTHmYy-fuqzSruyHd_J85eP36iDvF691g/edit?slide=id.g34f38648671_0_38#slide=id.g34f38648671_0_38)  
- **Project Report (PDF)**: [Download the report](https://drive.google.com/file/d/1qsoqbzCUXe_BxZ8T2tX2JuF5tg-JAVzi/view?usp=sharing)

---

