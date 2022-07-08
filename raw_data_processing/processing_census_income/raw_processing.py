import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

df = pd.read_table('raw_census_income',
                        delimiter=', '
                        , na_values='?').dropna()
tranfer_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'if<=50K']

for col in tranfer_cols:
    df[col] = df[col].astype('category').cat.codes

train, test_val = train_test_split(df, test_size=0.4)

test, valid = train_test_split(test_val, test_size=0.5)

np.savetxt('train', train.values, fmt='%d', delimiter='\t')
np.savetxt('test', test.values, fmt='%d', delimiter='\t')
np.savetxt('valid', valid.values, fmt='%d', delimiter='\t')