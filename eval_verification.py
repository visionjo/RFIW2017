#!/usr/bin/python
# Simple script to score outputs for kinship verification validation set as part of RFIW 2017.
# Results must be in a zip file with no parent directory (as is the case for actually submissions)
# email: robinson.jo@husky.neu.edu with any questions
import os
import numpy as np
import shutil
import zipfile

root_dir = os.path.dirname(os.path.realpath(__file__))

# type labels, which are also the filenames results should be named as
pair_types = ['bb', 'fd', 'fs', 'md', 'ms', 'sibs', 'ss']
output_filename = 'scores.txt'

# directory containing groundtruth for validation set (CSV files)
gtdir = root_dir + 'data/verification_p2/val/'
gtpaths = [gtdir + s + '_val.csv' for s in pair_types]

# path to zip folder container results
zippath = root_dir + '/resources/sample_submissions/verification/sample_submission_verification_val.zip'

# unzip results in temp directory
tmpdir = 'fiwtmp/'
zip_ref = zipfile.ZipFile(zippath, 'r')
os.makedirs(tmpdir, exist_ok=True)
zip_ref.extractall(tmpdir)
zip_ref.close()

# list of paths to all results
fpaths = [tmpdir + s + '.csv' for s in pair_types]

do_score = True

rpaths = [] # results paths (i.e., fpaths that exist)
for f in fpaths:
    if not os.path.exists(f):
        print('results file missing: ', f)
        do_score = False
    else:
        rpaths.append(f)

# check that whether results files were found
if not rpaths:
    # if no results found then terminate
    print('************************************************************')
    print('************************* ERROR ****************************')
    print('************************************************************')
    print('*** No results found. Exit()                             ***')
    exit()
if not do_score:
    # if only subset of results found give option to score some
    print('************************************************************')
    print('************************* WARNING **************************')
    print('************************************************************')
    print('*** Not all of the 7 sets of results were found.         ***')
    print('*** Results for all 7 types must be included in submission.*')
    print('************************************************************\n')
    print('Would you like to score the results found?',end='\t')
    response = 'n'
    response = input('y/[n]: ')
    if not response == 'y' and not response == 'yes':
        print('Exit()')
        exit()

print(rpaths)
scores = []
for k, f in enumerate(rpaths):
    # iterate over result files found
    clab = os.path.basename(f).replace('.csv','')
    ids = pair_types.index(clab)

    print('Scoring pair type ', pair_types[ids])

    # true labels
    tl = np.loadtxt(f, dtype=np.int, delimiter='\r\r\n')

    # results to score
    score = np.loadtxt(rpaths[k], dtype=np.int, delimiter='\r\r\n')

    class1 = np.asarray(np.where(tl == 1))
    class0 = np.asarray(np.where(tl == 0))

    thresh = np.sort(score)
    nthresh = score.size

    # check inputs
    if tl.size != nthresh:
        print('Invalid size')
        exit()

    for i in range(0, nthresh):
        if score[i] > 1 | score[i] < 0:
            print('Input not only 1s and 0s')
            exit()

    hit_rate = np.zeros((nthresh,), dtype=np.float)
    fa_rate = np.zeros((nthresh,), dtype=np.float)
    rec = np.zeros((nthresh,), dtype=np.float)

    for thi in range(1, nthresh):
        th = thresh[thi]
        # hit rate = TP/P
        hit_rate[thi] = np.sum(score[class1] >= th) / class1.size
        # fa rate = FP/N
        fa_rate[thi] = np.sum(score[class0] >= th) / class0.size
        rec[thi] = (np.sum(score[class1] >= th) + np.sum(score[class0] < th)) / nthresh

    # area under curve
    AUC = np.sum(np.multiply(abs(fa_rate[2:-1] - fa_rate[1:-2]), hit_rate[2:-1]))

    # equal error rate
    i1 = -1
    i2 = -1
    for i in range(nthresh - 1, 0, -1):
        if hit_rate[i] >= (1 - fa_rate[i]):
            i1 = i
    for i in range(nthresh - 1, 0, -1):
        if (1 - fa_rate[i]) >= hit_rate[i]:
            i2 = i

    EER = 1 - max(1 - fa_rate[i1], hit_rate[i2])

    print('Accuracy for ' + clab + ': ' + str(max(rec)))

    # append score to determine overall average
    scores.append(max(rec))

# determine average accuracy
n_sets_scored = len(scores)
overall_acc = np.sum(scores)/n_sets_scored
print('Average accuracy for ' + str(n_sets_scored) + ' out of 7 types is ' + str(overall_acc))


# delete temp directory zip file was uncompressed into
shutil.rmtree(tmpdir)