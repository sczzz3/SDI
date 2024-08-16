
import argparse
import numpy as np
import pandas as pd
import mne
import torch
from model import Net


def match_channel_names(channel_names, target_keywords):
    matched_channels = {}
    for key, keywords in target_keywords.items():
        matched = []
        for keyword in keywords:
            if key == 'EEG':
                matched = [ch for ch in channel_names if keyword in ch]
                if not matched:
                    matched = [ch for ch in channel_names if 'EEG' in ch]
                if not matched:
                    matched = [ch for ch in channel_names if 'EEG' in ch or 'EEG' in ch]
            elif key == 'ECG':
                matched = [ch for ch in channel_names if ('ECG' in ch or 'EKG' in ch) and keyword in ch]
                if not matched:
                    matched = [ch for ch in channel_names if 'ECG' in ch or 'EKG' in ch]
            elif key == 'EOG':
                matched = [ch for ch in channel_names if ('EOG' in ch or 'ROC' in ch or 'LOC' in ch) and keyword in ch]
                if not matched:
                    matched = [ch for ch in channel_names if 'EOG' in ch or 'ROC' in ch or 'LOC' in ch]
            elif key == 'EMG':
                matched = [ch for ch in channel_names if ('EMG' in ch or 'chin' in ch.lower()) and keyword in ch]
                if not matched:
                    matched = [ch for ch in channel_names if 'EMG' in ch or 'chin' in ch.lower()]
            if matched:
                matched_channels[key] = matched[0]
                break
        if not matched:
            matched_channels[key] = None
    return matched_channels

def read(data_file):

    raw_data = mne.io.read_raw_edf(data_file, preload=True, verbose=False)
    raw_data.resample(sfreq=100)

    channel_names = raw_data.ch_names

    # Define the target keywords and conditions
    targets = {
        'EEG': ['C4'],
        'EOG': ['R'],
        'ECG': ['R', 'EKG'],
        'EMG': ['R']
    }

    matched_channels = match_channel_names(channel_names, targets)

    # # Print the matched channels
    # for key, channel in matched_channels.items():
    #     print(f"{key} channel: {channel}")


    selected_channels = [ch for ch in matched_channels.values() if ch is not None]
    raw_selected = raw_data.copy().pick(selected_channels, verbose=False)

    epoch_duration = 30
    epochs = mne.make_fixed_length_epochs(raw_selected, duration=epoch_duration, preload=True, verbose=False)

    C4 = matched_channels['EEG']
    chin = matched_channels['EMG']
    EOGR = matched_channels['EOG']
    ECG = matched_channels['ECG']
    uV_signal = epochs.get_data(picks=[C4, chin, EOGR], units='uV')
    mV_signal = epochs.get_data(picks=[ECG], units='mV')

    combined_signal = []
    for epoch_idx in range(len(uV_signal)):
        combined_epoch = np.concatenate((uV_signal[epoch_idx], mV_signal[epoch_idx]), axis=0)
        combined_signal.append(combined_epoch)

    combined_signal = np.array(combined_signal)

    # Print the shape of the resulting data
    # print(f"Shape of the combined signal: {combined_signal.shape}")
    return combined_signal

def save_to_csv(data1, data2, output_file):

    data1_np = data1.numpy()
    data2_np = data2.numpy()
    df = pd.DataFrame({
        'Column1': data1_np,
        'Column2': data2_np
    })
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process an EDF file and extract EEG, ECG, EOG, and EMG signals.')
    parser.add_argument('data_file', type=str, help='Path to the EDF file')
    parser.add_argument('output_file', type=str, help='Path to the output CSV file.')
    args = parser.parse_args()

    psg = read(args.data_file)

    device = torch.device("cuda")

    model = Net().to(device)
    model.load_state_dict(torch.load('../weight/checkpoint.pt'))
    model.eval()

    with torch.no_grad():
        psg = torch.tensor(psg).float().to(device)
        logits, pred_nerm = model(psg)
        logits = torch.sigmoid(logits)
        depth = logits.squeeze(dim=-1).cpu()
        pred_nerm = torch.argmax(pred_nerm, dim=-1).cpu()

    save_to_csv(depth, pred_nerm, args.output_file)