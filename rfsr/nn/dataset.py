import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset

from rfsr import encode, awgn

BASE_DIR = Path(__file__).resolve().parent


# --- Synthetic in-memory dataset ---
class SyntheticLoRaDataset(Dataset):
    def __init__(self, oversampling=1, size=100, payload_length=16, downsampling=8, SF=12, BW=125e3):
        self.size = size  # unused, we generate on-demand

        self.payload_length = payload_length
        self.OSF = oversampling

        # Channel parameters
        self.snr_range = (10, -22)

        # LoRa parameters
        self.center_freq = 915e6
        self.sf = SF
        self.bw = BW
        print(f"BW={BW}")
        self.sample_rate = 2e6 / downsampling  # make it equivalent to the OTA: default: Fs=2e6  (default: downsampling /8 -> 0.25e6)
        self.src = 0
        self.dst = 1
        self.seqn = 7
        self.cr = 4
        self.enable_crc = 1
        self.implicit_header = 0
        self.preamble_bits = 8

        # determine signal length
        self.input_len = len(
            encode(self.center_freq, self.sf, self.bw, np.ones(self.payload_length, dtype=np.uint8), self.sample_rate,
                   self.src, self.dst, self.seqn, self.cr, self.enable_crc, self.implicit_header, self.preamble_bits))
        self.output_len = self.input_len * self.OSF

        # create the dataset
        x_tensors, y_tensors, snr_tensors = [], [], []
        print(f"Dataset size: {self.size}")
        for i in range(self.size):

            if i % 100 == 0:
                print(f"Added: {i} elements")

            # high-fs signal (oversampled by self.OSF) <- label
            y = encode(self.center_freq, self.sf, self.bw,
                       np.random.randint(0, 2 ** 8 - 1, size=self.payload_length, dtype=np.uint8),
                       self.sample_rate * self.OSF, self.src, self.dst, self.seqn, self.cr, self.enable_crc,
                       self.implicit_header, self.preamble_bits)

            # low-fs signal w/ noise <- input
            x = y[::self.OSF]  # extract down-sampled signal
            snr_db = np.random.uniform(*self.snr_range)
            x = awgn(x, snr_db)  # add noise

            # convert to tensor
            x_tensor = torch.tensor(np.stack([x.real, x.imag]), dtype=torch.float32)
            y_tensor = torch.tensor(np.stack([y.real, y.imag]), dtype=torch.float32)
            snr_tensor = torch.tensor([snr_db], dtype=torch.float32)

            x_tensors.append(x_tensor)
            y_tensors.append(y_tensor)
            snr_tensors.append(snr_tensor)  # Append inside the loop

        self.x_batched = torch.stack(x_tensors, dim=0)
        self.y_batched = torch.stack(y_tensors, dim=0)

        # store the SNR values
        self.snr_batched = torch.stack(snr_tensors, dim=0)  # Now size will be (size, 1)

    def __len__(self):  # size of the dataset
        return self.size

    def __getitem__(self, idx):
        return self.x_batched[idx], self.y_batched[idx], self.snr_batched[idx],
