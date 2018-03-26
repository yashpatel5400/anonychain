"""
__author__ = Yash Patel
__name__   = extract.py
__description__ = Extracts the information of the ownership of wallets from the format
provided in the Fistful of Bitcoin paper to that employed here. 
"""

import csv

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
    