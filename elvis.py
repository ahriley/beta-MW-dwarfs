import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *

Mpc_to_km = 3.086*10**19
km_to_kpc = 10**3/Mpc_to_km

#subs = pd.read_csv('data/elvis/Hall&Oates.txt', sep='\t')

with open('data/elvis/Hall&Oates.txt') as f:
    print(f.readline())
    print(f.readline())
    print(f.readline())
    print(f.readline())
subs
