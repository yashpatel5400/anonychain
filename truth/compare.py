"""
__author__ = Yash Patel
__name__   = extract.py
__description__ = Given classifications provided by the clustering algorithm (given in 
a binary pickle format), determines the corresponding accuracy 
"""

import pickle

def compute_accuracy(truth_fn, guess_fn):
    truth = pickle.load(open(truth_fn, "rb"))
    guess = pickle.load(open(guess_fn, "rb"))


if __name__ == "__main__":
    compute_accuracy(
        "output/SpectralClustering_guess.pickle", 
        "truth/extracted.pickle")