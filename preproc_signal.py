# ==================================================================
# File name: `preproc_signal.py`
# Created by: Nicholas Dominic <nicholas.dominic@binus.ac.id>
# Last edited by: Nicholas Dominic <nicholas.dominic@binus.ac.id>
# Last edited at: October 25th, 2023
# ------------------------------------------------------------------
# Note:
#   This is a proprietary file. Do NOT alter or use this code
#   without the author's written permission.
# <Copyright -2023, Jakarta, Indonesia>
# ==================================================================

import wfdb, heartpy, re, hrvanalysis as hrva
from biosppy.signals import ecg
from numpy import ndarray
from time import time as t
from datetime import datetime as dt

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
class SignalPreprocessing():
    def __init__(self, record_idx : str, db : str, *args, **kwargs) -> None:
        super(SignalPreprocessing, self).__init__()
        self.record_idx = record_idx
        self.db = db
        
        # record
        self.r = wfdb.rdrecord(record_name=self.record_idx, pn_dir=self.db)
        
        # annotation
        self.a = wfdb.rdann(record_name=self.record_idx, pn_dir=self.db, extension='atr')
        
        self.signals = self.r.p_signal # can be multi-dimensional channels
        self.channels = self.r.sig_name
        self.freq_rate = self.r.fs
        
    def extract_rpeaks(self, signal, *args, **kwargs) -> ndarray:
        # segment
        rpeaks, = ecg.engzee_segmenter(
            signal=signal,
            sampling_rate=self.freq_rate
        )

        # correct R-peak locations
        rpeaks, = ecg.correct_rpeaks(
            signal=signal, rpeaks=rpeaks,
            sampling_rate=self.freq_rate, tol=.05
        )

        # extract templates
        _, rpeaks = ecg.extract_heartbeats(
            signal=signal, rpeaks=rpeaks,
            sampling_rate=self.freq_rate, before=.2, after=.4
        )
        
        return rpeaks
        
    def filter_signal(self, signal, low_freq, high_freq, *args, **kwargs) -> ndarray:
        return heartpy.filter_signal(
            signal, filtertype='bandpass',
            cutoff=[low_freq, high_freq], sample_rate=self.freq_rate
        )
    
    def get_peaklist(self,
        start_partitions : list,
        duration : int,
        label : int,
        resampling : int = 0,
        *args, **kwargs
    ) -> dict:
        '''
        Example return:
        return_example = {
            "peaks" : {
                1000 : [
                    {"channel" : "ECG1", "value" : [20, 30, 40]},
                    {"channel" : "ECG2", "value" : [50, 60, 70]},
                ],
                2500 : [
                    {"channel" : "ECG1", "value" : [55, 66, 77]},
                    {"channel" : "ECG2", "value" : [75, 85, 96]},
                ]
            }, 
            "created_at" : "2024-02-15", 
            "exc_time" : 50.184
        }
        '''
        
        # to calculate exc. time (in seconds)
        start_time = t()
        
        # reassign variables since they can be overrided if resampling != 0
        fr = self.freq_rate
        signals = self.signals
        
        # resampling signal (if any)
        if resampling != 0:
            resampled_signals, _ = wfdb.processing.resample_multichan(
                self.signals, self.a, fr, resampling
            )
            fr = resampling
            signals = resampled_signals
            
            # if you did resampling, partitions index should be adjusted
            # based on the newest max. signal length.
            start_partitions = [i for i in start_partitions if i <= len(signals)]
        
        FINAL_RESULTS, PARTITION = {}, {}
        for start_partition in start_partitions:
            delta = (duration * 60) * fr # remember, freq = N / time(s), therefore N = freq x time
            curr_signals = signals[start_partition:start_partition+delta, :]
            
            PEAKS_CHANNEL = []
            for i, channel in enumerate(self.channels):
                signal = curr_signals[:, i]
                signal = self.filter_signal(signal, 7, 30) # filtered
                peaks = self.extract_rpeaks(signal)
                peaks = [int(p) for p in peaks] # convert from np.int32 to INT
                PEAKS_CHANNEL.append(dotdict({"channel" : channel, "value" : peaks}))
            PARTITION[int(start_partition)] = PEAKS_CHANNEL
        
        FINAL_RESULTS["peaks"] = PARTITION
        FINAL_RESULTS["label"] = label
        FINAL_RESULTS["exc_time"] = round(t()-start_time, 3)
        FINAL_RESULTS["created_at"] = dt.now().strftime("%Y-%m-%d %X")
        
        return dotdict(FINAL_RESULTS)

def get_record_start_partitions(
    record : str,
    db : str,
    min_duration : int,
    freq_rate : int,
    resampling : int = 0,
    max_total_partition : int = 10,
    *args, **kwargs
) -> dict:
    
    def get_label(annotation):
        p = re.compile("([A-Za-z]+)")
        return p.search(annotation)[1]
    
    start_sample = (min_duration * 60) * freq_rate # freq = N / time(s), therefore N = freq x time
    annotation = wfdb.rdann(record, 'atr', pn_dir=db, sampfrom=start_sample)
    signals, _ = wfdb.rdsamp(record, pn_dir=db, sampfrom=start_sample)
    
    if resampling != 0:
        start_sample = (min_duration * 60) * resampling
        signals, annotation = wfdb.processing.resample_multichan(
            signals, annotation, freq_rate, resampling)
    
    results = []
    if db.lower() == "vfdb":
        annotations = [get_label(a) for a in annotation.aux_note]
        annotations = list(map(lambda x: x.replace('NSR', 'N').replace('VFIB', 'VF'), annotations))
        positive_labels = ["VT", "VF", "VFL"]
        
        for ann, annot_sample in zip(annotations, annotation.sample):
            if ann in positive_labels:
                results.append(annot_sample-start_sample) # n sample before events
    
    elif db.lower() == "nsrdb":
        nsrdb_start_partition = 0
        
        for i in range(max_total_partition):
            results.append(nsrdb_start_partition)
            nsrdb_start_partition += start_sample
            
    else:
        raise ValueError("Available DB: 'vfdb' and 'nsrdb' only.")
    
    results = [r for r in results if r >= 0] # remove non-negative values
    return results

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