
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import mne

import xmltodict
import warnings
warnings.filterwarnings("ignore")


channel_alias = {
# 'C3': ['EEG(sec)', 'EEG 2', 'EEG2', 'EEG sec', 'EEG(SEC)'],
'C4': ['EEG'],
'ECG': ['ECG'],
# 'EOGL': ['EOG(L)'],
'EOGR': ['EOG(R)'],
'EMG': ['EMG'],
}

def extract(data_file, anno_file):

    visits = data_file.split('/')[-2].split('shhs')[1]
    record_id = data_file.split('/')[-1].split('.edf')[0].split('shhs'+visits+'-')[1]
    nsrr_dataset = pd.read_csv("../shhs/datasets/shhs-harmonized-dataset-0.20.0.csv")
    record_row = nsrr_dataset[(nsrr_dataset.loc[:, 'nsrrid']==int(record_id)) & (nsrr_dataset.loc[:, 'visitnumber']==int(visits))]
    NSRR_AHI = float(record_row.loc[:, ['nsrr_ahi_hp4u_aasm15']].values[0][0])

    raw_data = mne.io.read_raw_edf(data_file, preload=True, verbose=False)
    pick_channels = {}
    try:
        for chan in channel_alias.keys():
            pick_channels[chan] = list(set(channel_alias[chan]).intersection(set(raw_data.info.ch_names)))[0]
    except:
        with open("error_shhs.txt", 'a+') as f:
            f.write(data_file + '\t' + str(raw_data.info.ch_names) + '\n')
        return [], [], [], None

    raw_data = raw_data.pick(list(pick_channels.values()))
    raw_data.resample(sfreq=100)
    # sf = raw_data.info["sfreq"]

    # XML-NSRR type
    with open (anno_file, 'r') as f:
        xml_text = f.read()
    json_text = xmltodict.parse(xml_text)
    events_list = dict(json_text)['PSGAnnotation']["ScoredEvents"]['ScoredEvent']
    onset = []
    duration = []
    desc = []

    for event in events_list:
        desc.append(event["EventConcept"])
        onset.append(float(event["Start"]))  
        duration.append(float(event["Duration"]))

    anno_data = mne.Annotations(onset, duration, desc)     
    raw_data.set_annotations(anno_data, emit_warning=False, verbose=False)

    annotation_desc_2_event_id = {
                                'Wake|0': 0,
                                'Stage 1 sleep|1': 1,
                                'Stage 2 sleep|2': 2,
                                'Stage 3 sleep|3': 3,
                                'Stage 4 sleep|4': 3,
                                'REM sleep|5': 4,
                                }

    events_data, event_id_mapping = mne.events_from_annotations(
        raw_data, event_id=annotation_desc_2_event_id, 
        chunk_duration=30., 
        verbose=False,)

    tmax = 30. - 1. / raw_data.info['sfreq']

    event_id = {}
    if np.any(np.unique(events_data[:, 2] == 0)):
        event_id['Wake'] = 0
    if np.any(np.unique(events_data[:, 2] == 1)):
        event_id['Stage 1'] = 1
    if np.any(np.unique(events_data[:, 2] == 2)):
        event_id['Stage 2'] = 2
    if np.any(np.unique(events_data[:, 2] == 3)):
        event_id['Stage 3 / Stage 4'] = 3
    if np.any(np.unique(events_data[:, 2] == 4)):
        event_id['REM sleep'] = 4

    epochs_data = mne.Epochs(raw=raw_data, 
                            events=events_data,
                            event_id=event_id,
                            event_repeated='merge',
                            tmin=0., tmax=tmax, 
                            baseline=None, preload=True, verbose=False)

    ###
    # C3 = list(set(channel_alias['Fz-Cz']).intersection(set(raw_data.info.ch_names)))[0]
    C4 = list(set(channel_alias['C4']).intersection(set(raw_data.info.ch_names)))[0]      
    chin = list(set(channel_alias['EMG']).intersection(set(raw_data.info.ch_names)))[0]  
    ECG = list(set(channel_alias['ECG']).intersection(set(raw_data.info.ch_names)))[0]  
    # EOGL = list(set(channel_alias['EOGL']).intersection(set(raw_data.info.ch_names)))[0]  
    EOGR = list(set(channel_alias['EOGR']).intersection(set(raw_data.info.ch_names)))[0]  

    uV_signal = epochs_data.get_data(picks=[C4, chin, EOGR], copy=True, units='uV')
    mV_signal = epochs_data.get_data(picks=[ECG], copy=True, units='mV')
    psg_signal = np.concatenate((uV_signal, mV_signal), axis=1)

    ########################################################################
    sleep_events = {'Wake|0': 0, 'Stage 1 sleep|1': 1, 'Stage 2 sleep|2': 2, 'Stage 3 sleep|3': 3, 'Stage 4 sleep|4': 3, 'REM sleep|5': 4}
    apnea_events = {'Central apnea|Central Apnea': 1, 'Obstructive apnea|Obstructive Apnea': 1}
    hypopnea_events = {'Hypopnea|Hypopnea': 1}
    arousal_events = {'Arousal|Arousal ()': 1}


    EPOCH_LENGTH = 30
    epsilon = 1e-6

    total_epochs = int(raw_data.times[-1] // EPOCH_LENGTH) + 1
    excluded_epochs = set()
    epoch_sleep_stages = [None] * total_epochs  # Sleep stages per epoch
    epoch_apnea_events = [0] * total_epochs
    epoch_hypopnea_events = [0] * total_epochs  
    epoch_arousal_events = [0] * total_epochs  

    for event in events_list:
        event_type = event["EventType"]
        event_concept = event["EventConcept"]
        start_time = float(event["Start"])
        duration = float(event["Duration"])
        end_time = start_time + duration
        start_epoch = int(start_time // EPOCH_LENGTH)
        end_epoch = int((start_time + duration - epsilon) // EPOCH_LENGTH)

        if event_type == "Stages|Stages":
            if event_concept not in sleep_events:
                for epoch in range(start_epoch, min(end_epoch + 1, total_epochs)):
                    excluded_epochs.add(epoch)
            else:
                for epoch in range(start_epoch, min(end_epoch + 1, total_epochs)):
                    epoch_sleep_stages[epoch] = sleep_events[event_concept]

        for epoch in range(start_epoch, min(end_epoch + 1, total_epochs)):
            epoch_start_time = epoch * EPOCH_LENGTH
            epoch_end_time = (epoch + 1) * EPOCH_LENGTH
            
            overlap_start = max(start_time, epoch_start_time)
            overlap_end = min(end_time, epoch_end_time)
            overlap_duration = max(0, overlap_end - overlap_start)
            
            overlap_fraction = overlap_duration / EPOCH_LENGTH
            
            if event_type == "Respiratory|Respiratory" and event_concept in apnea_events:
                epoch_apnea_events[epoch] = max(epoch_apnea_events[epoch], overlap_fraction)

            elif event_type == "Respiratory|Respiratory" and event_concept in hypopnea_events:
                epoch_hypopnea_events[epoch] = max(epoch_hypopnea_events[epoch], overlap_fraction)

            elif event_type == "Arousals|Arousals" and event_concept in arousal_events:
                epoch_arousal_events[epoch] = max(epoch_arousal_events[epoch], overlap_fraction)


    sleep_stages = [stage for epoch, stage in enumerate(epoch_sleep_stages) if epoch not in excluded_epochs]
    apnea_events = [event for epoch, event in enumerate(epoch_apnea_events) if epoch not in excluded_epochs]
    hypopnea_events = [event for epoch, event in enumerate(epoch_hypopnea_events) if epoch not in excluded_epochs]
    arousal_events = [event for epoch, event in enumerate(epoch_arousal_events) if epoch not in excluded_epochs]

    epoch_sleep_stages_np = np.array(sleep_stages, dtype=int)
    epoch_apnea_events_np = np.array(apnea_events, dtype=float)
    epoch_hypopnea_events_np = np.array(hypopnea_events, dtype=float)
    epoch_arousal_events_np = np.array(arousal_events, dtype=float)
    epochs_labels = np.stack((epoch_sleep_stages_np, epoch_apnea_events_np, epoch_hypopnea_events_np, epoch_arousal_events_np), axis=-1)

    return psg_signal, epochs_labels, NSRR_AHI


if __name__ == '__main__':

    # cohort_list = ['shhs1', 'shhs2']
    cohort_list = ['shhs1']
    data_path = '../shhs/polysomnography/edfs/'
    label_path = '../shhs/polysomnography/annotations-events-nsrr/'
    save_path = '../proc_data/shhs/'
    os.makedirs(save_path, exist_ok=True)

    for cohort in cohort_list:

        cur_data_path = os.path.join(data_path, cohort)
        cur_label_path = os.path.join(label_path, cohort)

        records = os.listdir(cur_data_path)
        annos = [x.split('.')[0]+'-nsrr.xml' for x in records]

        for i in tqdm(range(len(records)), desc=cohort):

            if os.path.exists(os.path.join(save_path, records[i].split('.')[0]+'.npz')):
                continue

            if os.path.exists(os.path.join(cur_label_path, annos[i])):
                psg, oximetry, y_data, ahi = extract(os.path.join(cur_data_path, records[i]), os.path.join(cur_label_path, annos[i]))
            if len(psg) > 0:
                np.savez(os.path.join(save_path, records[i].split('.')[0]), psg=psg, oxi=oximetry, y=y_data, ahi=ahi)

