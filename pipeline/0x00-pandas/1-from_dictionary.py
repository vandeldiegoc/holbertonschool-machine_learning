#!/usr/bin/env python3
""" module """
import pandas as pd


alphabet_string = ['A', 'B', 'C', 'D']
init_dict = {'First': [0.0, 0.5, 1.0, 1.5],
         'Second': ['one', 'two', 'three', 'four']}

df = pd.DataFrame(init_dict, index=alphabet_string)
