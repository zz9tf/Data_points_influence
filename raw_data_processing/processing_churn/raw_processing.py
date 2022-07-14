import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

df = pd.read_csv('row_Churn_Modelling.csv',
                    delimiter=',',
                    header=0,
                    usecols=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
                )
tranfer_cols = ['Geography', 'Gender']

for col in tranfer_cols:
    df[col] = df[col].astype('category').cat.codes

train, test_val = train_test_split(df, test_size=0.4)

test, valid = train_test_split(test_val, test_size=0.5)

np.savetxt('train', train.values, fmt='%f', delimiter='\t')
np.savetxt('test', test.values, fmt='%f', delimiter='\t')
np.savetxt('valid', valid.values, fmt='%f', delimiter='\t')