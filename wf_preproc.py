import wfdb
from os import getcwd
from pandas import DataFrame as df

SAVE_FOLDER_DATA = getcwd() + "\\data"

def slpdb_mat_to_df(dataset, *args, **kwargs) -> None:
    ''' To convert .mat to Pandas Dataframe, and then save it as .csv file. '''
    
    filename = []
    time = []
    rr_interval = []
    annotation = []
    age = []
    gender = []
    weight = []

    for data in dataset:
        filename.append(data[0]["filename"][0])
        time.append(data[0]["time"][0])
        rr_interval.append(data[0]["rr"][0])
        annotation.append(data[0]["annotation"][0])
        age.append(data[0]["age"][0])
        gender.append(data[0]["gender"][0])
        weight.append(data[0]["weight"][0])

    df({"filename" : filename, "time" : time, "rr_interval" : rr_interval, \
        "annot" : annotation, "age" : age, "gender" : gender, "weight" : weight}) \
        .to_csv("data_OLD/SlpDbData.csv", index=False)
    
def get_db(key : int = None, *args, **kwargs) -> dict:
    ''' To get the WFDB database. '''
    
    database = wfdb.get_dbs()
    database_ = {d[0] : d[1] for d in database}
    
    if key is None:
        return database_
    else:
        selected_db = database_[key]
        print("DB_NAME: {}".format(selected_db))
        
        try:
            records = wfdb.get_record_list(db_dir=key)
            print("TOTAL_RECORDS: {}".format(len(records)))
        except Exception as e:
            records = ""
            print("TOTAL_RECORDS: {}".format(len(records)))
            print("[ERROR] {}".format(e))
        
        return {"db" : key, "records" : records}
    
def extract_rdheader(list_of_records : list, db : str, save_folder : str = SAVE_FOLDER_DATA, *args, **kwargs) -> None:
    ''' To extract the HEADER from all records. '''
    
    R_NAME, SIG_LEN, SAMPLING_FREQ, BASE_DATETIME = [], [], [], []
    for r in list_of_records:
        record = wfdb.rdheader(r, pn_dir=db)
            # channels == index of sig_name, for sleepdb, 0 : ECG/mV, 1: BloodPressure/mmHg, 2: EEG (C4-A1)/mV, 3: Resp (sum)/L
            # channels == index of sig_name, for mitdb, 0 : MLII/mV, 1: V5/mV

        R_NAME.append(record.record_name)
        SIG_LEN.append(record.sig_len)
        SAMPLING_FREQ.append(record.fs)

        default_base_date = "1990-01-04"
        default_base_time = "00:00:00"
        base_date = default_base_date if record.base_date is None else record.base_date.strftime("%Y-%m-%d")
        base_time = default_base_time if record.base_time is None else record.base_time.strftime("%X")
        BASE_DATETIME.append(base_date + " " + base_time)

    df({
        "r_name" : R_NAME,
        "sig_len" : SIG_LEN,
        "sampling_freq" : SAMPLING_FREQ,
        "created_at" : BASE_DATETIME
    }).to_csv("{}/{}-rdheader.csv".format(save_folder, db), index=False)
    
def show_ann_label(*args, **kwargs):
    return wfdb.show_ann_labels()
    
def extract_ann(list_of_records : list, db : str, ext : str = "st", save_folder : str = SAVE_FOLDER_DATA, *args, **kwargs) -> None:
    ''' To extract the ANNOTATION from all records. '''
    
    R_NAME, ANN_IDX, ANN_AUX = [], [], []
    for r in list_of_records:
        annot = wfdb.rdann(record_name=r, pn_dir=db, extension=ext)
        
        R_NAME.extend([r] * len(annot.sample))
        ANN_IDX.extend(annot.sample)
        ANN_AUX.extend([aux.split(" ")[0] for aux in annot.aux_note]) # remove non-sleep annotation

    df({
        "r_name" : R_NAME,
        "annot_idx" : ANN_IDX,
        "annot_aux" : ANN_AUX
    }).to_csv("{}/{}-rdann.csv".format(save_folder, db), index=False)