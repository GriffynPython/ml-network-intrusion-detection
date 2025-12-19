# Network Intrusion Detection System (IDS)

## Overview
A machine learning–based Network Intrusion Detection System (IDS) built using the NSL-KDD dataset.
The system classifies network traffic as **normal** or **attack** and exposes predictions via a FastAPI REST API.

## Dataset
- NSL-KDD dataset
- Binary classification: normal vs attack
- One-hot encoding for categorical features

## Model
- Gradient Boosting Classifier
- High accuracy on unseen test data
- Handles class imbalance effectively

## Results
- Accuracy: ~98%
- Precision / Recall: High for both classes
- Confusion matrix included in screenshots

## API Endpoints
- `GET /health` → service health check
- `POST /predict` → intrusion prediction

### Example Request
```json
{
  "features": {
    "count": 9,
    "dst_bytes": 5450,
    "duration": 0,
    "flag_SF": 1,
    "logged_in": 1,
    "protocol_type_tcp": 1,
    "same_srv_rate": 1,
    "service_http": 1,
    "src_bytes": 181,
    "srv_count": 9
  }
}
