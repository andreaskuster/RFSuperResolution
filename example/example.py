import torch
import numpy as np

from rfsr import decode, encode, awgn
from rfsr.nn import load_eval_model


def main():
    ######
    #  Load RFSR model
    model_name = "model_model0v0_bs5_osf4_ds250_lr0.001_wd1e-05"
    model = load_eval_model(f"{model_name}")

    ######
    # Generate LoRa Packet
    sf, bw, sample_rate = 12, 125e3, 2e6
    center_freq, src, dst, seqn, payload = 915e6, 0, 1, 7, np.random.randint(0, 255, size=16, dtype=np.uint8)
    signal = encode(center_freq, sf, bw, payload, sample_rate, src, dst, seqn, 4, 1, 0, 8)

    ######
    # Add noisy channel
    snr = -22 # np.random.uniform(-35, 10)
    signal = awgn(signal, snr)

    ######
    # Downsample signal to 0.25e6
    DS = 8  # DS to 0.25e6
    signal_ds = signal[::DS]

    ######
    # RFSR: Upsample and filter 4x to 1e6
    UPS = 4 # upsampling factor
    with torch.no_grad():
        iq_tensor = torch.tensor(
            np.stack([signal_ds.real, signal_ds.imag]),
            dtype=torch.float32
        ).unsqueeze(0)  # adds batch dim
        output = model(iq_tensor, snr)  # shape (1, 2, L*OSF)
    signal_nn = output[0, 0, :].cpu().numpy() + 1j * output[0, 1, :].cpu().numpy()

    ######
    # Evaluate packet error rate
    try:
        x = decode(signal_nn, sf, bw, (UPS * sample_rate) / DS)
    except:
        x = []
    if len(x) != 1:
        print(f"Number of decoded packets: {len(x)} != 1 (actual:{len(x)})-> Fail")
    else:
        print("Success")


if __name__ == "__main__":
    main()
