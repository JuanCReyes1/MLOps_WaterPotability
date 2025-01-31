import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score,precision_score, recall_score,f1_score



def load_data(filepath: str) -> pd.DataFrame:
    #test_data = pd.read_csv('./data/processed/test_processed.csv')
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Encountered an error loading data from: {e}")
    

def prepare_data(data : pd.DataFrame) -> tuple[pd.DataFrame,pd.Series]:
    try:
        X = data.drop(columns=["Potability"],axis=1)
        y = data["Potability"]
        return X,y
    
    except Exception as e:
        raise Exception(f"Error preparing data: {e}")
    

def load_model(filepath:str):
    #model = pickle.load(open("model.pkl","rb"))
    try:
        with open(filepath,"rb") as file:
            model = pickle.load(file)
        return model
    
    except Exception as e:
        raise Exception(F"Error loading model from {filepath}:{e}")
    


def evaluation_model(model,X_test:pd.DataFrame,y_test:pd.Series) -> dict:

    try:
        
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test,y_pred)
        pre = precision_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        f1 = f1_score(y_test,y_pred)

        metrics_dict = {

            'acc':acc,
            'precision':pre,
            'recall': recall,
            'f1_score': f1

        }
        #print(f"Model evaluation metrics: {metrics_dict}")  # Debugging line
        return metrics_dict

    except Exception as e:
        raise Exception(f"Encountered an error evaluating the model: {e}")
    
def save_metrics(metrics : dict, metrics_path : str) -> None:
    try:
        with open(metrics_path,'w') as file:
            json.dump(metrics, file,indent=4)

    except Exception as e:
        raise Exception(f"Error saving metrics to {metrics_path}: {e}")
    


def main():

    try:
        test_data_path = "./data/processed/test_processed.csv"
        model_path = "./model.pkl"
        metrics_path = "./metrics.json"

        test_data = load_data(test_data_path)
        X_test, y_test = prepare_data(test_data)
        model = load_model(model_path)
        metrics = evaluation_model(model,X_test,y_test)
        save_metrics(metrics,metrics_path)
    except Exception as e:
        raise Exception(f"An Error Occurred: {e}")
    

if __name__ == "__main__":
    main()





    
