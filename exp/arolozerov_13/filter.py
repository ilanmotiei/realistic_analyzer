import json
import pandas as pd

with open('a.json', 'r', encoding='utf-8') as f:
    data = json.load(fp=f)

df = pd.DataFrame.from_records(data['data'])
df['DEALDATE'] = pd.to_datetime(df['DEALDATESTRING'], format='%d.%m.%Y')
relevant = df[(df['DEALDATE'].dt.year >= 2023) & (df['ASSETMETER'] <= 90) & (df['ASSETMETER'] >= 50) & (df['ROOMNUM'] <= 4) & (df['ROOMNUM'] >= 2)]
relevant.sort_values(by='DEALS_DEALAMOUNT').to_csv('relevant.csv')
