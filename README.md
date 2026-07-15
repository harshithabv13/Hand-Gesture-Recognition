# Hand Gesture Recognition Application Using Thermal Imaging

A web-based computer vision application built using Python, Flask, and TensorFlow. The application utilizes a fine-tuned MobileNet architecture to recognize hand gestures in real-time, leveraging thermal/infrared signatures to ensure robust performance across diverse environmental conditions.

---

## 🚀 Features
* **Real-Time Recognition:** High-accuracy gesture prediction powered by a MobileNet architecture.
* **Thermal Image Optimization:** Built to process thermal inputs for superior detection reliability.
* **Web-Based Interface:** Clean HTML5/CSS3 frontend designed for smooth presentation displays.
* **Secure User Management:** Local database tracking user authentication seamlessly.

---

## 💡 Why Thermal Imaging?
We opted for thermal/infrared data structures over traditional visual RGB inputs to solve real-world computer vision vulnerabilities:
* **Total Illumination Independence:** Flawless tracking in absolute darkness, shifting shadows, or intense glare because the system relies entirely on heat emission rather than visible light.
* **Natural Background Subtraction:** Eliminates complex visual noise (like room clutter or moving objects) by naturally isolating high-contrast human body heat signatures against cooler ambient backgrounds.
* **Privacy-First Computer Vision:** Intuitively reads structural hand contours and gestures without capturing high-resolution facial features or sensitive personal identity data.
* **Low Latency Processing:** Thermal matrices reduce feature complexity, allowing the MobileNet model to achieve high frame-rate processing efficiency even on resource-constrained hardware.

---

## 📂 Project Structure
```text
├── Dataset/          # Contains thermal images utilized for training evaluation
├── models/           # Stores the trained deep learning weight models (mobilenet.h5)
├── src/              # Core application logic
│   ├── app.py        # Main Flask execution server
│   └── templates/    # Frontend web UI layouts (Login, Registration, Dashboard)
├── static/           # UI design assets, style sheets, and visualization plots
├── uploads/          # Temporary directory for image evaluation streams
└── requirements.txt  # Project framework dependencies

#🛠️ Technology Stack
Backend Framework: Flask (Python)

Deep Learning Engine: TensorFlow / Keras

Computer Vision Processing: OpenCV

Model Architecture: MobileNet

#💻 Setup and Installation
Prerequisites
Make sure you have Python 3.8+ installed on your computer.

2. Install Dependencies
Install all required package libraries using pip:

pip install -r requirements.txt

3. Run the Application
Start the local development server:

python src/app.py
Open your web browser and navigate to http://127.0.0.1:5000 to interact with the application dashboard.
