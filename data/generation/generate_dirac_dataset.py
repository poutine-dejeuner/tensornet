import torch
import pandas as pd
import math
from copy import deepcopy


def generate_dirac_dataset(n_samples, min_len, max_len, num_channels=5):

    # Initialize lists
    samples_feats = []
    samples_amplitudes = []
    samples_distances = []

    # Generate sets of random numbers for amplitude, frequency and phase
    if max_len > min_len:
        all_len = torch.randint(high=max_len - min_len, size=[n_samples]) + min_len
    else:
        all_len = torch.ones(n_samples, dtype=int) * max_len

    all_factors = torch.rand(size=(n_samples, num_channels, 2))

    # Generate 2 Diracs for each sample, for each channel
    for ii in range(n_samples):
        this_len = all_len[ii]

        amplitudes, feats, distances = [], [], []
        for channel in range(num_channels):
            pos = torch.randperm(this_len)[:2]
            amp = torch.rand(1) + 0.2
            dist = abs(pos[0] - pos[1])
            feat = torch.zeros(this_len, dtype=torch.float32)
            feat[pos] = amp

            amplitudes.append(amp)
            feats.append(feat)
            distances.append(dist)

        # append the features and labels to the lists
        samples_feats.append(torch.stack(feats, dim=0))
        samples_amplitudes.append(torch.tensor(amplitudes))
        samples_distances.append(torch.tensor(distances))

    # Stack lists into a 2D tensor
    samples_feats = torch.stack(samples_feats, dim=0)
    samples_amplitudes = torch.stack(samples_amplitudes, dim=0)
    samples_distances = torch.stack(samples_distances, dim=0)

    # Generate dictionaries of labels
    dict_features = {f'feat_{ii}': samples_feats[:, :, ii].tolist() for ii in range(samples_feats.shape[2])}
    dict_amplitudes = {f'amplitude_{ii}': samples_amplitudes[:, ii] for ii in range(samples_amplitudes.shape[1])}
    dict_distances = {f'distance_{ii}': samples_distances[:, ii] for ii in range(samples_distances.shape[1])}
    dict_labels = deepcopy(dict_features)
    dict_labels.update(dict_amplitudes)
    dict_labels.update(dict_distances)

    # Generate pandas DataFrame with the features and labels
    df = pd.DataFrame()
    df = df.assign(**dict_labels)

    return df


if __name__ == "__main__":
    torch.random.manual_seed(111)
    min_len = 30
    max_len = 30
    num_channels = 5
    df = generate_dirac_dataset(n_samples=10000, min_len=min_len, max_len=max_len, num_channels=num_channels)
    df.to_csv(f'data/dirac_dataset_len-{min_len}-{max_len}_num-channels-{num_channels}.csv', index=False)
    print(df)

    print('Done!')



