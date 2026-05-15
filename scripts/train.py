"""Compatibility entry point for training.

The active training implementation follows the hoangnv/g layout in
``src/custom_train.py``; this wrapper is kept for old commands.
"""
from src.custom_train import main

if __name__ == "__main__":
    main()
