#!/usr/bin/python

import re
import os
import sys
from filecmp import cmp
import pandas as pd
import numpy as np

def is_needed(filename, same_parameters):
    for param in same_parameters.split('-'):
        if param not in filename:
            return False
    return True

if __name__ == "__main__":
    root_path = os.path.abspath('..')
    raw_results = root_path + '/results/raw_results'
    script_results = root_path + '/results/script_results'

    
    tasks = ['/gender_results/',
             '/icc_results/BNU_1/',
             '/icc_results/HNU_1/',
             '/icc_results/IPCAS_1/',
             '/pairwise_results/']

    for i, task in enumerate(tasks):
        print(i, task)

    print('\nPlease choose what directories to check:')

    choosen = np.array([int(x) for x in input().split()])
    
    for choice in choosen:
        assert choice >= 0 and choice <= 4

    tasks = np.take(tasks, choosen)

    for current_task in tasks:

        if not os.path.exists(script_results + current_task):
            os.makedirs(script_results + current_task)

        print('{}\n'.format(current_task))
        
        needed_files = os.listdir(raw_results + current_task)
        num_of_needed_files = len(needed_files)

        current_files = os.listdir(script_results + current_task)
        num_of_curren_files = len(current_files)
        num_of_not_similar = 0

        for f in current_files:
            if f in needed_files:

                raw_df    = pd.read_csv(raw_results    + current_task + f)
                script_df = pd.read_csv(script_results + current_task + f)

                assert (raw_df.columns.values == script_df.columns.values).all()

                for column in raw_df.columns.values:
                    if 'best_params' not in column and 'icc_values' not in column:
                        assert (raw_df[column] == script_df[column]).all()

                    if 'best_params' in column:
                        for idx in raw_df.index:
                            assert eval(raw_df[column][idx]) == eval(script_df[column][idx])

                    if 'icc_values' in column:
                        for idx in raw_df.index:
                            raw_column = re.sub('nan, ', '', raw_df[column][idx])
                            script_column = re.sub('nan, ', '', script_df[column][idx])
                            assert eval(raw_column) == eval(script_column)
            else:
                num_of_curren_files -= 1
                num_of_not_similar  += 1
                print('File {} not from needed results.'.format(f))

        print('\n\t{}/{} are similar'.format(num_of_curren_files, num_of_needed_files))
        print('\t{} ain\'t similar\n'.format(num_of_not_similar))
