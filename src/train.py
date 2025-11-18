import pickle
import pandas as pd

from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder

from xgboost import XGBClassifier


def load_data():
    df = pd.read_csv('data/heart.csv')
    return df


def train(df):
    X_train = df[['ST_Slope', 'ChestPainType', 'ExerciseAngina', 'Oldpeak', 'MaxHR']]
    y_train = df['HeartDisease']

    cat_col = ['ST_Slope', 'ChestPainType', 'ExerciseAngina']
    num_col = ['Oldpeak', 'MaxHR']

    preprocessor = make_column_transformer(
        (OrdinalEncoder(), cat_col),
        ('passthrough', num_col)
    )

    pipeline = make_pipeline(
        preprocessor,
        XGBClassifier(min_child_weight=10,max_depth=5, eta=0.001, seed=42)
    )

    pipeline.fit(X_train, y_train)

    return pipeline


def save_model(pipeline, outputfile):
    with open(outputfile, 'wb') as f_out:
        pickle.dump(pipeline, f_out)

df = load_data()
pipeline = train(df)
save_model(pipeline, 'model/model.bin')