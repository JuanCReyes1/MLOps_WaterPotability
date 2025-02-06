import pandas as pd
import numpy as np
import os
import pickle
import yaml
from sklearn.ensemble import RandomForestClassifier

def load_params(params_path : str) -> int:
    #n_estimators = yaml.safe_load(open("params.yaml","r"))["model_building"]["n_estimators"]
    try:
        with open(params_path,"r") as file:
            params = yaml.safe_load(file)
        return params["model_building"]["n_estimators"]
    
    except Exception as e:
        raise Exception(f"Error loading parameters from {params_path}: {e}")
    

def load_data(filepath : str) -> pd.DataFrame:
    #train_data = pd.read_csv('./data/processed/train_processed.csv')
    try:
        return pd.read_csv(filepath)
    
    except Exception as e:
        raise Exception(f"Error loading data from {filepath} : {e}")
    

def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame,pd.Series]:
    try:
        X = data.drop(columns=['Potability'],axis=1)
        y = data['Potability']
        return X,y
    except Exception as e:
        raise Exception(f"Error Preparing Data: {e}")

def train_model(X: pd.DataFrame, y : pd.Series, n_estimators : int) -> RandomForestClassifier:
    try:
        clf = RandomForestClassifier(n_estimators=n_estimators)
        clf.fit(X,y)
        return clf 
    except Exception as e:
        raise Exception(f"Error training model: {e}")
    

def save_model(model: RandomForestClassifier, filepath: str) -> None:
    try:
        with open(filepath, "wb") as file:
            print(filepath)
            pickle.dump(model,file)
    except Exception as e:
        raise Exception(f"Error saving the model: {e}")

def main():
    try:
        params_path = "params.yaml"
        data_path = "./data/processed/train_processed.csv"
        model_name = "./models/model.pkl" 

        n_estimators = load_params(params_path)
        train_data = load_data(data_path)
        X_train, y_train = prepare_data(train_data)
        model = train_model(X_train, y_train, n_estimators)
        save_model(model, model_name)
    except Exception as e:
        raise Exception(f"Encountered an error: {e}")

if __name__ == "__main__":
    main()
