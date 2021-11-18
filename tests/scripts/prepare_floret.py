"""
This script adds a small set of floret vectors for testing.
"""

import floret


if __name__ == "__main__":
    model = floret.train_unsupervised("tests/data/foobar.txt")
    model.save_model("tests/floret_vectors.bin")
    print("local model saved for floret")
