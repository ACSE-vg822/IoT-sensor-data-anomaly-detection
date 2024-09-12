# IoT-sensor-data-anomaly-detection

This repository contains a **Streamlit** app for detecting anomalies in IoT sensor data using an LSTM-based machine learning model. The app allows users to upload sensor data, process it, and visualize detected anomalies against the training data. https://iot-sensor-data-anomaly-detection-zqbd3yzrtsnnhnwhbm4bcs.streamlit.app/

## Features
- Upload a custom CSV file or use the default dataset provided.
- Visualize IoT sensor data, including **Temperature**, **Humidity**, **Air Quality**, **Light**, and **Loudness**.
- Highlight detected anomalies in the sensor data using a machine learning model.
- Download detected anomalies in CSV format.
- Interactive and clean UI with **progress bars**, **sidebar navigation**, and **custom styling**.

## Tech Stack
- **Streamlit**: For creating the web interface.
- **TensorFlow/Keras**: For training and running the LSTM anomaly detection model.
- **Matplotlib**: For plotting sensor data and anomalies.
- **Pandas**: For data manipulation and CSV handling.

## Installation

### Prerequisites
- Python 3.8 or higher
- Git installed on your local machine

### Clone the repository
```bash
git clone https://github.com/your-username/anomaly-detection-iot.git
cd anomaly-detection-iot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Project Structure
```bash
anomaly-detection-iot/
│
├── src/                         
│   ├── pipeline/
│   │   └── predict_pipeline.py   
│   └── components/ 
│       ├── data_ingestion.py 
│       ├── data_transformation.py
│       └── model_trainer.py         
├── artifacts/                    
│   ├── model.pkl
│   ├── train.csv
│   ├── test.csv                
│   └── data.csv                  
│
├── streamlit_app.py               
├── requirements.txt
├── setup.py             
└── README.md                      
```
Feel free to open issues and contribute to this project!