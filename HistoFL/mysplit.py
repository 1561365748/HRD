import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.model_selection import train_test_split
k = 5

def split_train_test_valid():
    # read file
    input_path = "./dataset_csv/"
    file = "classification_hrd_dataset_fl.csv"
    # file = "classification_hrd_dataset.csv"
    df = pd.read_csv(input_path + file)

    for index in range(k):
        df_tmp = df
        train, valid, test =[],[],[]
        # df_col = pd.DataFrame()
        # df_col.columns = ['train', 'val', 'test']
        for site_id in range(2):
            site_str = str(site_id)
            df_tmp['institute'] = df_tmp['institute'].str.slice(-1)
            df_cur = df_tmp[df_tmp['institute'] == site_str]
            df_cur = df_cur['slide_id']
            # define the ratios 8:1:1
            train_len = int(len(df_cur) * 0.8)
            test_len = int(len(df_cur) * 0.1)
            # split the dataframe
            idx = list(df_cur.index)
            np.random.shuffle(idx)  # 将index列表打乱
            train += df_cur.loc[idx[:train_len]].values.tolist()
            test += df_cur.loc[idx[train_len:train_len + test_len]].values.tolist()
            valid += df_cur.loc[idx[train_len + test_len:]].values.tolist() # 剩下的就是valid
            # dimen = np.array(train).shape
            # print("train", dimen)
            # dimen = np.array(test).shape
            # print("test:", dimen)
            # dimen = np.array(valid).shape
            # print("valid:", dimen)

        df_col = DataFrame([train, valid, test]).T
        # print(df_col)
        df_col.columns = ['train', 'val', 'test']
        # site_id = df['institute'][0][-1]
        df_col.to_csv("./splits/fl_classification/splits_{}.csv".format(index))
        # df_col.to_csv("./splits/nofl_classification/splits_{}.csv".format(index))

split_train_test_valid()
