import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from musubi_tuner.flux_kontext_distill_train import main

if __name__ == "__main__":
    main()
