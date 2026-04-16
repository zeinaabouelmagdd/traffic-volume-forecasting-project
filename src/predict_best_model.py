import pandas as pd

def predict_with_model(model, X_new: pd.DataFrame):
    predictions = model.predict(X_new)
    return predictions