# ==================================================================
# File name: `step4_get_live_prediction.py`
# Created by: Nicholas Dominic <nicholas.dominic@binus.ac.id>
# Last edited by: Nicholas Dominic <nicholas.dominic@binus.ac.id>
# Last edited at: November 03rd, 2023
# ------------------------------------------------------------------
# Note:
#   This is a proprietary file. Do NOT alter or use this code
#   without the author's written permission.
# <Copyright -2023, Jakarta, Indonesia>
# ==================================================================

import pickle, os, json, shutil, hrvanalysis as hrva
from scipy.stats import entropy
from argparse import ArgumentParser
from pandas import read_csv as rcsv
from numpy import array
from time import time as t
from datetime import datetime as dt
from sklearn.preprocessing import MinMaxScaler as mms

def rr_preproc(rr_interval : list) -> list:
    nn_interval = hrva.remove_outliers(rr_intervals=rr_interval, verbose=False)

    # @param method: "malik", "kamath", "karlsson", "acar"
    nn_interval = hrva.remove_ectopic_beats(rr_intervals=nn_interval, method="malik", verbose=False)

    # @param interpolation_method: 'linear', 'time', 'index', 'values', 'nearest', 'zero', 'slinear',
    # 'quadratic', 'cubic', 'barycentric', 'krogh', 'spline', 'polynomial', 'from_derivatives',
    # 'piecewise_polynomial', 'pchip', 'akima', 'cubicspline'
    nn_interval = hrva.interpolate_nan_values(rr_intervals=nn_interval, interpolation_method="cubic")

    # remove NaN values which weren't filtered during interpolation; e.g., in the last index
    nn_interval = [i for i in nn_interval if str(i) != "nan"]
    
    return nn_interval

def extract_ftr(nn_interval : list, *args, **kwargs) -> dict:
    FEATURES = {}
    nn_interval = rr_preproc(nn_interval)

    # TIME DOMAIN
    ftr_time_domain = hrva.get_time_domain_features(nn_interval)
    FEATURES.update(ftr_time_domain)

    ftr_geometric_time_domain = hrva.get_geometrical_features(nn_interval)
    FEATURES.update(ftr_geometric_time_domain)

    # Frequency Domain
    ftr_freq_domain = hrva.get_frequency_domain_features(nn_interval)
    FEATURES.update(ftr_freq_domain)

    # Non-linear Domain
    ftr_entropy = hrva.get_sampen(nn_interval) # sample entropy
    # entropy_val = ftr_entropy["sampen"]
    entropy_val = entropy(nn_interval) # to avoid entropy value becomes INF
    FEATURES.update({"entropy" : entropy_val})

    ftr_poincare = hrva.get_poincare_plot_features(nn_interval)
    FEATURES.update(ftr_poincare)

    # CVI (Cardiac Sympathetic Index), CSI (Cardiac Vagal Index)
    ftr_csi_cvi = hrva.get_csi_cvi_features(nn_interval)
    FEATURES.update(ftr_csi_cvi)
    
    return FEATURES

def start_prediction(model, samples : list, *args, **kwargs) -> list:
    # data preprocessing and feature preparation
    scaler = mms()
    selected_ftr = ['mean_nni', 'rmssd', 'mean_hr', 'triangular_index', \
                    'total_power', 'csi', 'cvi', 'entropy']

    FINAL_RESULT = []
    PREDICTION_FTRS = []

    for i in samples:
        result = {"submission_id" : i["submission_id"]}
        all_features = extract_ftr(i["rr_interval_ms"])
        result.update(all_features)
        FINAL_RESULT.append(result)

        input_features = list({key : all_features[key] for key in selected_ftr}.values())
        PREDICTION_FTRS.append(input_features)

    # model prediction
    # print(PREDICTION_FTRS)
    input_scaled = scaler.fit_transform(array(PREDICTION_FTRS))
    prediction = model.predict(input_scaled).tolist()

    # save results (to ../completion/ folder)
    for f, p, s in zip(FINAL_RESULT, prediction, samples):
        desc = "malignant_ventricular_ectopy" if p == 0 else "normal_sinus_rhythm"
        f.update({
            "prediction_label" : int(p),
            "prediction_desc" : desc,
            "src_created_at" : s["created_at"],
            "prc_dt" : dt.now().strftime("%Y-%m-%d %X")
        })
    
    return FINAL_RESULT

if __name__ == "__main__":
    # example how to run:
    # - for Linux --> <PY_VENV>/bin/python [FILENAME].py -M "svm"
    # - for Windows --> <PY_VENV>\Scripts\python [FILENAME].py -T "svm"
    parser = ArgumentParser()
    parser.add_argument('-M', '--model', type=str, required=True, help="Select the model for prediction.", choices=["lr", "svm", "rf", "mlp"])
    parser.add_argument('-U', '--use-sample', type=bool, required=False, help="If you don't have any data yet, set use-sample=True.", default=False)
    
    args = vars(parser.parse_args())
    USE_SAMPLE = args["use_sample"]
    SELECTED_MODEL = args["model"]
    
    # map
    model_map = {
        "lr" : "logistic_regression",
        "svm" : "support_vector_machine",
        "rf" : "random_forest",
        "mlp" : "multiple_layer_perceptron"
    }
    
    start_time = t()
    print("[WARNING] Do not close your terminal. Otherwise, the process will be terminated.")
    print("[INIT] Commencing training using '{}' model ...".format(model_map[SELECTED_MODEL]))
    
    # get samples
    samples = []
    if USE_SAMPLE:
        submission_id = "Polar_H10_C77D752D_20230922_201243_RR"
        queues = ["polar_h10_data/{}.txt".format(submission_id)]
        load_sample = rcsv(queues[0], sep=";")
        samples.append({
            "submission_id" : submission_id,
            "rr_interval_ms" : load_sample.rr_interval_ms.tolist(),
            "created_at" : "2023-01-01 00:00:00"
        })
    else:
        queue_path = "polar_h10_data/queue/"
        queues = [queue_path+p for p in os.listdir(queue_path) if p.endswith(".json")]
        for queue_filepath in queues:
            with open(queue_filepath) as f:
                samples.append(json.loads(f.read()))
    print("[DONE] Found {} file(s) to be processed.".format(len(samples)))

    # load the latest model
    model_path = "models/{}/".format(SELECTED_MODEL)
    latest_pkl = sorted([p for p in os.listdir(model_path) if p.endswith(".pkl")], reverse=True)[0]
    model = pickle.load(open(model_path + latest_pkl, 'rb'))
    print("[DONE] Model `{}` was successfully loaded.".format(model))

    # get prediction
    my_results = start_prediction(model, samples)
    print("[COMPLETED in {:.3f}s] Prediction results:".format(t()-start_time), end="\n\n")
    for file, r in zip(queues, my_results):
        print(">> File path: {}".format(file))
        print(r, end="\n\n")

    # move from ../queue/ to ../archive/
    if not USE_SAMPLE:
        for q in queues:
            shutil.move(src=q, dst="polar_h10_data/archive/")
        print("[COMPLETED] All data from ./queue/ was successfully moved to ./archive/")