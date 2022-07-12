import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

df = pd.read_csv('row_ratio.csv',
                    delimiter=',',
                    header=0
                )

train, test_val = train_test_split(df, test_size=0.4)

test, valid = train_test_split(test_val, test_size=0.5)

np.savetxt('train', train.values, fmt='%f', delimiter='\t')
np.savetxt('test', test.values, fmt='%f', delimiter='\t')
np.savetxt('valid', valid.values, fmt='%f', delimiter='\t')