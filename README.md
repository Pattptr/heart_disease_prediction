# Heart disease prediction

## Problem Description
Heart disease remains one of the leading causes of death worldwide. Early detection and risk assessment are crucial for effective prevention and treatment. This project develops a machine learning model to predict the likelihood of heart disease based on clinical data. To help identify high-risk individuals who may benefit from early medical intervention.

## Prerequisites

- Python 3.8 or higher
- uv package manager

## Installation

1. Clone the repository:
   
```bash
git clone https://github.com/Pattptr/heart_disease_prediction.git
cd heart_disease_prediction
```
2. (macOS users only) Install OpenMP for XGBoost:

```bash
brew install libomp
```

3. Install dependencies using uv:
   
```bash
uv sync
```

## Running the Project

### Training the Model

```bash
uv run python train.py
```


### Running Jupyter Notebooks

```bash
uv run jupyter notebook
```
### Running the FastAPI Server
To start the prediction API server:

```bash
uv run uvicorn src.predict:app --reload
```

The API will be available at http://localhost:8000. You can access the interactive API documentation at http://localhost:8000/docs.

### Running with Docker

If you prefer to use Docker:

```bash
docker build -t heart-disease-prediction .
docker run -p 8888:8888 heart-disease-prediction
```

## Project Structure

```
├── data/                      
│   └── heart.csv              # Heart disease dataset
├── model/                     
│   └── model.bin              # Trained machine learning model
├── src/                       
│   ├── __init__.py            # Package initialization
│   ├── predict.py             # FastAPI prediction endpoint
│   └── train.py               # Model training script
├── heart_disease_pred.ipynb   # Main prediction notebook
├── Dockerfile                 # Docker configuration
├── pyproject.toml             # Project dependencies (uv)
├── uv.lock                    # Locked dependencies
├── .python-version            # Python version specification
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

## Dataset

This project used [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) from Kaggle, which can be found in the data folder in this repository, or  downloaded it directly from Kaggle.
### Feature Descriptions

| Feature | Description | Values/Range |
|---------|-------------|--------------|
| Age | Age of the patient | Years |
| Sex | Sex of the patient | M: Male, F: Female |
| ChestPainType | Chest pain type | TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic |
| RestingBP | Resting blood pressure | mm Hg |
| Cholesterol | Serum cholesterol | mm/dl |
| FastingBS | Fasting blood sugar | 1: if FastingBS > 120 mg/dl, 0: otherwise |
| RestingECG | Resting electrocardiogram results | Normal: Normal, ST: ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: Left ventricular hypertrophy by Estes' criteria |
| MaxHR | Maximum heart rate achieved | Numeric value between 60 and 202 |
| ExerciseAngina | Exercise-induced angina | Y: Yes, N: No |
| Oldpeak | ST depression induced by exercise relative to rest | Numeric value measured in depression |
| ST_Slope | Slope of the peak exercise ST segment | Up: upsloping, Flat: flat, Down: downsloping |
| HeartDisease | Output class (target variable) | 1: heart disease, 0: Normal |

