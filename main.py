import pandas as pd
from Preprocess import *

if __name__ == '__main__':
    data_frame = pd.read_excel(r'Dataset\train.xlsx')
    #data_frame = pd.read_excel(r'Dataset\evaluation.xlsx')

    data_frame = data_frame.dropna(axis=1)
    print("Data Frame Description: \n")
    print(data_frame.describe())

    main(data_frame)