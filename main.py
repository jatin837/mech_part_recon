#!/home/dj4t9n/dev/mech_part_recon/env/bin/python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # To avoid tensorflow warning

import tensorflow as tf


import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="Path to file")

args = vars(ap.parse_args())

path = os.path.abspath(args["path"])

print(path)

