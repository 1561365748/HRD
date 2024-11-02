import os
import csv

l = []
for filename in os.listdir(r'DATA_ROOT_DIR/classification_features_dir/h5_files'):
    l.append(filename)

f = open("name.csv", "w")
for line in l:
    f.write(line + '\n')
f.close()
