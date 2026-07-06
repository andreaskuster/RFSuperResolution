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



class OTALoRaDataset(Dataset):

    def __init__(self, oversampling = 1,
                 snr_range = (-20, 20), #(-35, 20),
                 training=True,
                 trim=False,
                 return_snr=False,
                 downsampling=8):
        self.return_snr = return_snr
        self.training = training
        self.trim = trim
        self.OSF = oversampling
        self.DSF = downsampling
        # we sample at 2MSPS IQ by default (oversampling of 8x)
        # we need to downsample the training data accordingly
        # /8 -> 0.25e6
        # /4 -> 0.5e6
        # /2 -> 1e6
        # /1 -> 2e6

        # load dataset parameters
        # --- Configuration ---
        DATA_DIRECTORY = "deepstudy/nn/data"
        min_snr, max_snr = snr_range
        final_list, self.snrs = filter_files_by_snr(DATA_DIRECTORY, min_snr, max_snr)
        # ---------------------

        # map them to the post processed, fulltrim files
        # self.files = list(map(lambda x: x.replace("ota", "post").replace(".cfile", "_fulltrim.cfile"), final_list))
        self.files = list(map(lambda x: x.replace("data/ota", f"{BASE_DIR}/data/post").replace(".cfile", "_fulltrim.cfile"), final_list))


        # map the files to the reference fiels
        def filename2refnum(filename):
            # 1. Get the base filename: 'exp1_000020_rxg18_0.cfile'
            basename = os.path.basename(filename)
            # 2. Split the name by the underscore:
            #    ['exp1', '000020', 'rxg18', '0.cfile']
            parts = basename.split('_')
            # 3. Get the second item (at index 1)
            extracted_id = parts[1]
            return f"{int(extracted_id)%100:06d}"

        self.refmap = list(map(lambda x: filename2refnum(x), self.files))

        self.size = len(self.files)
        print(f"OTALoRaDataset contains {self.size} items.")

        # split into training (80%) and testing (20%), random
        self.train_indices, self.test_indices = get_reproducible_split(self.size, test_split=0.2)


    def __len__(self):
        # return self.size
        if self.training:
            return len(self.train_indices)
        else:
            return len(self.test_indices)


    def __getitem__(self, idx):

        if self.training:
            idx = self.train_indices[idx]
        else:
            idx = self.test_indices[idx]

        # load signal
        signal = np.fromfile(self.files[idx], dtype=np.complex64)

        if self.trim:
            # Perform the trimming using array slicine
            TRIM_SIZE = 460_000
            signal = signal[TRIM_SIZE: -TRIM_SIZE]

        # downsample
        signal = signal[::self.DSF] # load 2MSPS OTA signal

        # extract the sample number
        ref_filename = os.path.join(BASE_DIR, "data/post/", f"signalout_{self.refmap[idx]}_fulltrim.cfile")
        ref_signal = np.fromfile(ref_filename, dtype=np.complex64) # load 2MSPS reference signal (label)

        # downsample to the right rate:
        if self.DSF/self.OSF < 1:
            raise RuntimeError(f"Invalid DSF/OSF ratio: {self.DSF}/{self.OSF}, needs to be >=1")
        ref_signal = ref_signal[::int(self.DSF/self.OSF)]

        if self.trim:
            ref_signal = ref_signal[TRIM_SIZE: -TRIM_SIZE]

        # Convert to (2, L) float tensor: channel 0 = real, 1 = imag
        x = torch.tensor(
            np.stack([signal.real, signal.imag], axis=0),
            dtype=torch.float32
        )
        y = torch.tensor(
            np.stack([ref_signal.real, ref_signal.imag], axis=0),
            dtype=torch.float32
        )

        if self.return_snr:
            return x, y, self.snrs[idx] # also return the snr
        else:
            return x, y


