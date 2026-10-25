"""Frozen-model calibration entry; keep the official VAE deterministic encoder path."""
# WanTokenizer identifies full-model training from a '/train.py' command-line path.
from train import main


if __name__ == '__main__':
    main()
