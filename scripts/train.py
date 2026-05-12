"""Entry point for training — called by DeepSpeed launcher via scripts/train.sh."""
from p2vg.train.train import main

if __name__ == "__main__":
    main()
