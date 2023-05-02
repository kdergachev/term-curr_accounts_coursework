import pandas as pd
import os
from datetime import datetime

data = pd.read_excel('new_data.xlsx', index_col=0, names=['holidays'])

data = data['holidays'].dropna()

data = pd.DataFrame(data)

data[5] = data.index.map(lambda x: int((x.day == 5) & (x.month != 1)))

for i in [10, 15, 20, 25]:
    data[i] = data.index.map(lambda x: int(x.day == i))

data['NY'] = data.index.map(lambda x: int((x.month == 12) & (x.day == 31)))


def carry_back(df, hol, ind):
    df = df.copy()
    h = False
    for i in range(len(df) - 1, -1, -1):
        
        if df[[hol, ind]].iloc[i, :].prod() == 1:
            continue
        elif df[ind].iloc[i] == 1:
            h = True
            df[ind].iloc[i] = 0
        elif h & (df[hol].iloc[i] == 1):
            df[ind].iloc[i] = 1
            h = False
    return df[ind]


for i in [5, 10, 15, 20, 25, 'NY']:
    data[i] = carry_back(data, 'holidays', i)

data['lower'] = 0
data['higher'] = 0

data.to_excel('data_with_indicators.xlsx')
"""
orig = pd.read_excel('current_accounts.xlsx', index_col=0)

orig.to_excel(f'current_accounts_archive_{str(datetime.now())[:10]}.xlsx')

orig.loc[orig.index.isin(data.index), :] = data.loc[data.index.isin(orig.index)]
orig = pd.concat([orig, data.loc[~data.index.isin(orig.index)]])

orig.to_excel('current_accounts.xlsx')

os.remove('new_data.xlsx')
"""