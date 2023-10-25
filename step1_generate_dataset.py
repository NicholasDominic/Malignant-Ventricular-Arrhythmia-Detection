# ==================================================================
# File name: `step1_generate_dataset.py`
# Created by: Nicholas Dominic <nicholas.dominic@binus.ac.id>
# Last edited by: Nicholas Dominic <nicholas.dominic@binus.ac.id>
# Last edited at: October 25th, 2023
# ------------------------------------------------------------------
# Note:
#   This is a proprietary file. Do NOT alter or use this code
#   without the author's written permission.
# <Copyright -2023, Jakarta, Indonesia>
# ==================================================================

import json, os
from argparse import ArgumentParser
from preproc_wfdb import get_db, extract_rdheader
from preproc_signal import get_record_start_partitions
from preproc_signal import SignalPreprocessing
from pandas import read_csv as rcsv
from datetime import datetime as dt

def call_db_info(key : str, label : int, *args, **kwargs) -> dict:
    """
    Available DB key options:
        * key="vfdb" for MIT-BIH Malignant Ventricular Ectopy Database
        * key="nsrdb" for MIT-BIH Normal Sinus Rhythm Database
    """
    
    dataset = get_db(key=key, verbose=False)
    RDHEADER_PATH = "data/{}-rdheader.csv".format(key)
    
    try:
        header = rcsv(RDHEADER_PATH)
    except:
        print("[ERROR] FileNotFoundError: No header data found.")
        print("Creating {} file ...".format(RDHEADER_PATH))
        h = extract_rdheader(list_of_records=dataset["records"], db=key)
        h.to_csv(RDHEADER_PATH, index=False)
        print("[COMPLETE] {} is successfully created.".format(RDHEADER_PATH))
    
    return {
        "db" : dataset["db"],
        "records" : dataset["records"],
        "header" : header,
        "label" : label
    }

def create_dataset(verbose : bool = True, *args, **kwargs) -> None:
    db = kwargs["db"]
    records = kwargs["records"]
    label = kwargs["label"]
    
    for r in records:
        print(" -- Retrieve start partition index for ID={}".format(r))
        st = get_record_start_partitions(record=r, db=db, min_duration=5, freq_rate=128)
        
        print(" -- Get signal peaks from each starting index")
        proc = SignalPreprocessing(record_idx=r, db=db)
        peaklist_result = proc.get_peaklist(start_partitions=st, duration=5, label=label)
        
        timestamp = dt.now().strftime("%Y%m%d")
        PATH = "data/{}_{}".format(timestamp, db)
        if not os.path.exists(PATH):
            os.mkdir(PATH)
        
        with open("{}/{}.json".format(PATH, r), "w") as outfile: 
            json.dump(peaklist_result, outfile)
        
        if verbose:
            print("[COMPLETED] Record {} (label={}) is successfully saved.".format(r, label))
    
if __name__ == "__main__":
    # example how to run:
    # - for Linux --> <PY_VENV>/bin/python [FILENAME].py -K "vfdb" -L 1
    # - for Windows --> <PY_VENV>\Scripts\python [FILENAME].py -K "vfdb" -L 1
    parser = ArgumentParser()
    parser.add_argument('-K', '--key', type=str, required=True, help="Waveform Database keys.", choices=["vfdb", "nsrdb"])
    parser.add_argument('-L', '--label', type=int, required=True, help="Label (or class) for the dataset.")
    
    args = vars(parser.parse_args())
    THIS_KEY = args["key"]
    THIS_LABEL = args["label"]
    
    print("[WARNING] Do not close your terminal. Otherwise, the process will be terminated.")
    print("[INIT] Creating {} dataset ...".format(THIS_KEY.upper()))
    create_dataset(**call_db_info(key=THIS_KEY, label=THIS_LABEL))
    print("[COMPLETED] Now you can proceed to `step2_generate_features.py`")