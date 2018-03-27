"""
__author__ = Yash Patel
__name__   = extract.py
__description__ = Extracts the information of the ownership of wallets from the format
provided in the Fistful of Bitcoin paper to that employed here. 
"""

import numpy as np
import pandas as pd
import pickle

def extract_from_raw(fn):
    """
    Data-format expected in the raw-data csv:
    1st column : transaction hash;
    2nd column : public key index on the sender side
    3rd column : sender public key
    4th column : label for that key
    5th column : public key index on the receiver side
    6th column : receiver public key
    7th column : label for that key.  
    """
    df = pd.read_csv(fn)
    columns = ["tx_hash","sender_id","sender_key","sender_label",
        "receiver_id","receiver_key","receiver_label"]
    label_to_wallets = {}

    senders   = np.unique(df["sender_label"])
    receivers = np.unique(df["receiver_label"])

    for sender in senders:
        sender_txs = df[df["sender_label"] == sender]
        label_to_wallets[sender] = set(sender_txs["sender_key"])

    for receiver in receivers:
        receiver_txs = df[df["receiver_label"] == receiver]

        # special case when dealing with change addresses to associate them back to 
        # the original person who made the transaction
        if "change" in receiver:
            np_receiver_txs = np.array(receiver_txs)
            for receiver_tx in np_receiver_txs:
                sender = receiver_tx[3]
                receiver_key = receiver_tx[5]
                if sender not in label_to_wallets:
                    label_to_wallets[sender] = { receiver_key }
                else:
                    label_to_wallets[sender].add(receiver_key)

        else:
            receiver_keys = set(receiver_txs["receiver_key"])
            if receiver not in label_to_wallets:
                label_to_wallets[receiver] = receiver_keys
            else:
                label_to_wallets[receiver] = label_to_wallets[receiver].union(receiver_keys)
    return label_to_wallets

def write_extracted(fn):
    label_to_wallets = extract_from_raw(fn)
    pickle.dump(label_to_wallets, open("truth/extracted.pickle","wb"))
    with open("truth/extracted.txt", "w") as f:
        for person in label_to_wallets:
            f.write("{} : {}\n".format(person, label_to_wallets[person]))

if __name__ == "__main__":
    write_extracted("truth/our_txs.csv")