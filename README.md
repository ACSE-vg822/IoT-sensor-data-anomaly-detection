# IoT-sensor-data-anomaly-detection

This repository contains a **Streamlit** app for detecting anomalies in IoT sensor data using an LSTM-based machine learning model. The app allows users to upload sensor data, process it, and visualize detected anomalies. https://iot-sensor-data-anomaly-detection-zqbd3yzrtsnnhnwhbm4bcs.streamlit.app/

## References
Data source: https://github.com/hkayann/grove-dataset-generation
Description as taken from the source:
- Contains humidity, temperature, light, loudness, and air quality data in order.
- Environment is 25 m2 studio room contains 2 people.
- Data is collected from 10/03/2021 18:36 PM to 11/03/2021 18.36 PM.
- Data might be considered as normal, there are no anomalies created on purpose.
- The groveHighAccTempDataset contains timestamp + temperature data. Environment is the same.

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
git clone https://github.com/your-username/IoT-sensor-data-anomaly-detection.git
cd IoT-sensor-data-anomaly-detection-iot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Project Structure
```bash
IoT-sensor-data-anomaly-detection/
│
├── src/                         
│   ├── pipeline/
│   │   └── predict_pipeline.py   
│   └── components/ 
│   │    ├── data_ingestion.py 
│   │    └── model_trainer.py 
│   ├── exception.py
│   └── logger.py
│
├── artifacts/                    
│   ├── model_new.keras             
│   └── data.csv                  
├── streamlit_app.py               
├── requirements.txt
├── setup.py             
└── README.md                      
```
Feel free to open issues and contribute to this project!
