import torch
import pandas as pd
import math
from copy import deepcopy


def generate_cosine_dataset(n_samples, min_len, max_len, num_frequencies=3):

    # Initialize lists
    samples_feats = []
    samples_amplitudes = []
    samples_frequencies = []
    samples_wavelengths = []

    # Generate sets of random numbers for amplitude, frequency and phase
    if max_len > min_len:
        all_len = torch.randint(high=max_len - min_len, size=[n_samples]) + min_len
    else:
        all_len = torch.ones(n_samples, dtype=int) * max_len

    all_factors = torch.rand(size=(n_samples, num_frequencies, 3))

    # Generate a coside for each sample
    for ii in range(n_samples):
        this_len = all_len[ii]
        
        # Get the random wavelengths, frequencies and phases
        wavelengths_factor = (all_factors[ii, :, 0])
        sort_idx = torch.argsort(wavelengths_factor, descending=False)
        wavelengths = wavelengths_factor[sort_idx] * 2 * (this_len - 1) + 1
        frequencies = 2 * math.pi / wavelengths
        amplitudes = all_factors[ii, sort_idx, 1]
        phases = all_factors[ii, sort_idx, 2] * 2 * math.pi

        # Generate the cosine signal
        x = float('nan') * torch.ones(max_len)
        x[:this_len] = torch.arange(this_len)
        x = x.unsqueeze(1)
        signal = amplitudes * torch.cos(frequencies * x + phases)
        signal = torch.sum(signal, dim=1)

        # append the features and labels to the lists
        samples_feats.append(signal)
        samples_amplitudes.append(amplitudes)
        samples_frequencies.append(frequencies)
        samples_wavelengths.append(wavelengths)

    # Stack lists into a 2D tensor
    samples_feats = torch.stack(samples_feats, dim=0)
    samples_amplitudes = torch.stack(samples_amplitudes, dim=0)
    samples_frequencies = torch.stack(samples_frequencies, dim=0)
    samples_wavelengths = torch.stack(samples_wavelengths, dim=0)

    # Generate dictionaries of labels
    dict_features = {f'feat_{ii}': samples_feats[:, ii] for ii in range(samples_feats.shape[1])}
    dict_amplitudes = {f'amplitude_{ii}': samples_amplitudes[:, ii] for ii in range(samples_amplitudes.shape[1])}
    dict_frequencies= {f'frequency_{ii}': samples_frequencies[:, ii] for ii in range(samples_frequencies.shape[1])}
    dict_wavelengths = {f'wavelength_{ii}': samples_wavelengths[:, ii] for ii in range(samples_wavelengths.shape[1])}
    dict_labels = deepcopy(dict_features)
    dict_labels.update(dict_amplitudes)
    dict_labels.update(dict_frequencies)
    dict_labels.update(dict_wavelengths)

    # Generate pandas DataFrame with the features and labels
    df = pd.DataFrame()
    df = df.assign(**dict_labels)
    
    return df


if __name__ == "__main__":
    torch.random.manual_seed(111)
    df = generate_cosine_dataset(n_samples=10000, min_len=30, max_len=30, num_frequencies=3)
    df.to_csv('data/cosine_dataset_len30.csv', index=False)
    print(df)

    print('Done!')




