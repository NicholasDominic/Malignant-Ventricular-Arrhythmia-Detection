# ==================================================================
# File name: `step2_generate_features.py`
# Created by: Nicholas Dominic <nicholas.dominic@binus.ac.id>
# Last edited by: Nicholas Dominic <nicholas.dominic@binus.ac.id>
# Last edited at: October 25th, 2023
# ------------------------------------------------------------------
# Note:
#   This is a proprietary file. Do NOT alter or use this code
#   without the author's written permission.
# <Copyright -2023, Jakarta, Indonesia>
# ==================================================================

import json, os, hrvanalysis as hrva
from preproc_signal import rr_preproc
from heartpy import analysis
from pandas import DataFrame as df
from argparse import ArgumentParser

if __name__ == "__main__":
    # example how to run:
    # - for Linux --> <PY_VENV>/bin/python [FILENAME].py -T "20231025"
    # - for Windows --> <PY_VENV>\Scripts\python [FILENAME].py -T "20231025"
    parser = ArgumentParser()
    parser.add_argument('-T', '--timestamp', type=str, required=True, help="Dataset folder timestamp, example: '20231025'.")
    
    args = vars(parser.parse_args())
    THIS_TIMESTAMP = args["timestamp"]
    
    print("[WARNING] Do not close your terminal. Otherwise, the process will be terminated.")
    print("[INIT] Preparing feature extraction processes ...")
    
    # retrieve dataset folder paths
    MAIN_PATH = "data/"
    dataset_folders = [p+"/" for p in os.listdir(MAIN_PATH) if p.startswith(THIS_TIMESTAMP)]
    print("[DONE] Retrieving dataset folders path: {}".format(dataset_folders))

    # check if datasets are available
    wfdb_keys = ["vfdb", "nsrdb"] # add here for more WFDB keys
    reports = {}
    
    for key in wfdb_keys:
        is_exist = []
        for i in dataset_folders:
            is_exist.append(key in i)
        reports[key] = any(is_exist)
    print("[REPORT] Is the required dataset available? {}".format(reports))
    
    check = [k for k, v in reports.items() if not v]
    if len(check) > 0:
        raise FileNotFoundError("[ERROR] You need to provide the dataset for: {} with timestamp={}".format(check, THIS_TIMESTAMP))
    
    # check if the features were extracted in THIS_TIMESTAMP
    SAVE_PATH = "feature_store/{}_features.csv".format(THIS_TIMESTAMP)
    if os.path.isfile(SAVE_PATH):
        print("[COMPLETED] Extracted features for timestamp={} already exists in this path: ./{}".format(THIS_TIMESTAMP, SAVE_PATH))
    else:
        # start extracting
        print("[INIT] Extracting features ...")
        ALL_FILES = []
        for folder in dataset_folders:
            list_files = [MAIN_PATH+folder+file for file in os.listdir(MAIN_PATH+folder) if file.endswith(".json")]
            ALL_FILES.extend(list_files)

        DF_RECORDS = []
        for filepath in ALL_FILES: # LOOP per file
            with open(filepath) as file:
                data = json.load(fp=file)

            print(" -- Processing file: {}".format(filepath))
            for idx in data["peaks"].keys(): # LOOP per start_partition
                for ch_num, ch in enumerate(data["peaks"][idx]): # LOOP per channel
                    peaklist = ch["value"]

                    try:
                        rr = analysis.calc_rr(peaklist=peaklist, sample_rate=128)
                        nn_interval = rr_preproc(rr_interval=rr["RR_list"])

                        FEATURES = {
                            "record_id" : int(filepath.split("/")[-1].split(".json")[0]),
                            "start_partition_idx" : idx,
                            "channel" : ch["channel"] + "_{}".format(str(ch_num))
                        }

                        # Reference:
                        # 1. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5624990/
                        # 2. https://aura-healthcare.github.io/hrv-analysis/hrvanalysis.html

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
                        FEATURES.update({"entropy" : ftr_entropy["sampen"]})

                        ftr_poincare = hrva.get_poincare_plot_features(nn_interval)
                        FEATURES.update(ftr_poincare)

                        # CVI (Cardiac Sympathetic Index), CSI (Cardiac Vagal Index)
                        ftr_csi_cvi = hrva.get_csi_cvi_features(nn_interval)
                        FEATURES.update(ftr_csi_cvi)

                        FEATURES.update({"label" : int(data["label"])}) # set label
                        DF_RECORDS.append(FEATURES)

                    except Exception as e:
                        print("File: {}\nError: {}".format(filepath, e))

        df(DF_RECORDS).to_csv(SAVE_PATH, index=False)
        print("[DONE] Saving dataframe to {}".format(SAVE_PATH))
    
    print("[COMPLETED] Now you can proceed to `step3_train_model.py`")