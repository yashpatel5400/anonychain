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
    
    truth_wallets = truth.values()
    guess_wallets = guess.values()

    guess_performance = {}
    for guess_wallet in guess_wallets:
        total_overlap = 0
        for true_owner in truth:
            overlap = len(guess_wallet.intersection(truth[true_owner]))
            total_overlap += overlap
            if overlap > 0:
                if guess_wallet not in guess_performance:
                    guess_performance[guess_wallet] = {}
                guess_performance[guess_wallet][true_owner] = overlap
        if total_overlap > 0:
            for owner in guess_performance[guess_wallet]:
                guess_performance[guess_wallet] /= total_overlap
    return guess_performance

def write_performance(truth_fn, guess_fn):
    guess_performance = compute_accuracy(truth_fn, guess_fn)
    with open("truth/accuracy.txt", "w") as f:
        for person in guess_performance:
            f.write("{} : {}\n".format(person, guess_performance[person]))

if __name__ == "__main__":
    write_performance(
        "output/SpectralClustering_guess.pickle", 
        "truth/extracted.pickle")