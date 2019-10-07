import os
import csv
import re
from collections import defaultdict
rapper_dict = set()
with open('rapworld.csv') as table:
    for line in table:
        end = line.find(';')
        rapper = line[0:end]
        rapper_dict.add(rapper)

with open('rappers.txt','w') as ofile:
    for rapper in rapper_dict:
        ofile.write(rapper + "\n")