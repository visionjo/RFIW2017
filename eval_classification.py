#!/usr/bin/env python
# Simple script to score output for family classification validation set as part of RFIW 2017.
# Results must be in a zip file with no parent directory (as is the case for actually submissions)
# email: robinson.jo@husky.neu.edu with any questions

import os
import numpy as np


root_dir = os.path.expanduser('~') + '/WORK/kinship/Data_Challenge/codalabs/'

f_results = root_dir + '/sample_submissions/classification/results.csv'
f_gt = root_dir + 'data/classification_p2/val/labels_val.csv'
output_filename = 'scores2.txt'


# true labels
tl = np.loadtxt(f_gt, dtype=np.int, delimiter='\r\r\n')
# results (output of run)
responses = np.loadtxt(f_results, dtype=np.int, delimiter='\r\r\n')

n_labels = len(tl)
n_response = len(responses)

if n_labels != n_response:
    # check number of labels matches that of responses
    print('Error: Number of responses does not match of ground truth labels')
    print('Number of labels in ground truth file: ', n_labels)
    print('Number of labels in results file: ', n_response)
    exit

output_file = open(output_filename, 'w')
acc = float((tl == responses).sum() / n_labels)

output_file.write('Accuracy: %f' % acc)
print("Accuracy: ", acc)
output_file.close()