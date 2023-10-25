# ==================================================================
# File name: `step3_train_model.py`
# Created by: Nicholas Dominic <nicholas.dominic@binus.ac.id>
# Last edited by: Nicholas Dominic <nicholas.dominic@binus.ac.id>
# Last edited at: October 25th, 2023
# ------------------------------------------------------------------
# Note:
#   This is a proprietary file. Do NOT alter or use this code
#   without the author's written permission.
# <Copyright -2023, Jakarta, Indonesia>
# ==================================================================

import cpuinfo, json, pickle, os
from pandas import read_csv as rcsv
from numpy import ndarray
from argparse import ArgumentParser
from time import time as t
from datetime import datetime as dt

from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import MinMaxScaler as mms
from sklearn.metrics import accuracy_score, auc, roc_curve, confusion_matrix as cfmat

from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


def evaluate(y_pred : ndarray, y_test : ndarray, *args, **kwargs) -> dict:
    start_time = t()
    
    tn, fp, fn, tp = cfmat(y_pred, y_test).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = (2 * precision * recall) / (precision + recall)
    fpr, tpr, threshold = roc_curve(y_test, y_pred)
    
    return {
        "accuracy" : float(accuracy_score(y_pred, y_test)),
        "true_pos" : int(tp),
        "true_neg" : int(tn),
        "false_pos" : int(fp),
        "false_neg" : int(fn),
        "recall" : float(recall),
        "precision" : float(precision),
        "f1_score" : float(f1_score),
        "false_pos_rate" : fpr.tolist(),
        "true_pos_rate" : tpr.tolist(),
        "auc_roc_threshold" : threshold.tolist(),
        "auc_roc_score" : float(auc(fpr, tpr)),
        "exc_time" : round(t()-start_time, 3),
        "created_at" : dt.now().strftime("%Y-%m-%d %X")
    }

if __name__ == "__main__":
    # example how to run:
    # - for Linux --> <PY_VENV>/bin/python [FILENAME].py -T "20231025"
    # - for Windows --> <PY_VENV>\Scripts\python [FILENAME].py -T "20231025"
    parser = ArgumentParser()
    parser.add_argument('-T', '--timestamp', type=str, required=True, help="Dataset folder timestamp, example: '20231025'.")
    parser.add_argument('-M', '--model', type=str, required=True, help="Select model for training.", choices=["lr", "svm", "rf", "mlp"])
    
    args = vars(parser.parse_args())
    THIS_TIMESTAMP = args["timestamp"]
    THIS_MODEL = args["model"]
    print("[WARNING] Do not close your terminal. Otherwise, the process will be terminated.")
    
    # load dataset
    DATASET_PATH = "feature_store/{}_features.csv".format(THIS_TIMESTAMP)
    my_dataset = rcsv(DATASET_PATH)
    print("[DONE] Load dataset: {}".format(DATASET_PATH))
    
    # set random state (seed)
    RAND_SEED = 43
    
    # feature selection
    selected_ftr = ['mean_nni', 'rmssd', 'mean_hr', 'triangular_index', \
                    'total_power', 'csi', 'cvi', 'entropy']
    X = my_dataset[selected_ftr]
    y = my_dataset.label
    print("[DONE] Features are selected: {}".format(selected_ftr))
    
    # split dataset
    X_train, X_test, y_train, y_test = tts(X, y, test_size=.3, random_state=RAND_SEED)
    
    # scale the data
    scaler = mms()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)
    print("[DONE] Split and scale the dataset using MinMaxScaler()")
    
    # map
    model_map = {
        "lr" : "logistic_regression",
        "svm" : "support_vector_machine",
        "rf" : "random_forest",
        "mlp" : "multiple_layer_perceptron"
    }
    
    # list of models
    model_selections = {
        "lr" : LR(random_state=RAND_SEED),
        "rf" : RFC(random_state=RAND_SEED),
        "svm" : SVC(random_state=RAND_SEED),
        "mlp" : MLPClassifier(
            hidden_layer_sizes=(16, 4),
            batch_size=64,
            learning_rate_init=1e-2,
            early_stopping=True,
            random_state=RAND_SEED
        )
    }
        
    # start training
    print("[INIT] Commencing training using '{}' model ...".format(model_map[THIS_MODEL]))
    start_training_time = t()
    model_id = str(int(t())) + "-{}".format(THIS_MODEL)
    
    model = model_selections[THIS_MODEL]
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    model_training_log = {
        "model_id" : model_id,
        "model_name" : model_map[THIS_MODEL],
        "model_version" : 1.0,
        "cloud_storage_uri" : "gs://",
        "evaluation" : evaluate(y_pred, y_test),
        "device_type" : "cpu",
        "device_name" : cpuinfo.get_cpu_info()['brand_raw'],
        "device_count" : 1,
        "exc_time_sec" : t()-start_training_time,
        "data_snapshot_dt" : THIS_TIMESTAMP,
        "prc_dt" : dt.now().strftime("%Y-%m-%d %X")
    }
    
    # save the model with in the pickle format
    SAVE_PATH = "models/{}".format(THIS_MODEL)
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    filepath = SAVE_PATH + "/{}_{}.pkl".format(dt.now().strftime("%Y%m%d"), THIS_MODEL)
    pickle.dump(model, open(filepath, 'wb'))
    print("[DONE] Saving {} model to {}".format(model_id, SAVE_PATH))
    
    # save the model log in the JSON format
    MODEL_LOG_PATH = "model_training_logs/{}".format(THIS_MODEL)
    if not os.path.exists(MODEL_LOG_PATH):
        os.mkdir(MODEL_LOG_PATH)
    
    with open("{}/{}.json".format(MODEL_LOG_PATH, model_id), "w") as outfile: 
        json.dump(model_training_log, outfile)
    print("[DONE] Saving the model log to {}".format(MODEL_LOG_PATH))
    
    print("[COMPLETED] Now you can reload the model from {} and use it for production purposes.".format(filepath))