"""
This script adds a small spaCy model that is used in testing.
"""

import fasttext


if __name__ == "__main__":
    model = fasttext.train_unsupervised("tests/data/foobar.txt", model="cbow", dim=10)
    model.save_model("tests/custom_fasttext_model.bin")
    print("local model saved for fasttext")
