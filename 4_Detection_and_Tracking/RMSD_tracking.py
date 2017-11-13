import os
import pandas as pd
import openpyxl
import sys
import itertools
import numpy as np
import cv2
import math
from shutil import copyfile

# Use read_tracking_file to read in the sample outputs and put them into a list
# Call read_compare_tracking_file to compute RMSD value

def read_tracking_file(path):
    tracking = []

    with open(path) as f:
        for line in f:
            newLine = []
            data = line.split(',')

            for item in data:
                newLine.append(int(item))
            tracking.append(newLine)

    return tracking

def read_compare_tracking_file(path, master, questionID):

    lines = len(master)
    tracking = []
    diff = 0

    with open(path) as f:
        for line in f:
            newLine = []
            data = line.split(',')

            for item in data:
                if (item[0] == 'x' or item[0] == 'y'):
                    newLine.append(int(item[2:len(item)]))
                else:
                    newLine.append(int(item))
            tracking.append(newLine)

    if (len(tracking) <= 0):
        return 0

    for i in range(0, len(tracking), 1):
        line_index = tracking[i][0]
        diff = diff + (master[line_index][1] - tracking[i][1]) * (master[line_index][1] - tracking[i][1]) + (master[line_index][2] - tracking[i][2]) * (master[line_index][2] - tracking[i][2])
    diff = diff / len(tracking)
    diff = math.sqrt(diff)

    score = 0;
    if (questionID == 0 or questionID == 1 or questionID == 3):
        score = (1 - diff / 100) * 33.333333
    else:
        score = (1 - diff / 100) * 20

    score = score - math.fabs(lines - len(tracking)) * 2;
    score = max(score, 0)

    return score