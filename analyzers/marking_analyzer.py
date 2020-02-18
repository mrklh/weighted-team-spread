import pandas as pd
import random
import numpy as np

xl_file = pd.ExcelFile("Transmittance.xlsx")

dfs = {sheet_name: xl_file.parse(sheet_name)
          for sheet_name in xl_file.sheet_names}

data = dfs['Transmittance'].values

rands1 = []
rands2 = []
rands3 = []
rands4 = []

for i in range(10):
    rands1.append(random.random()*random.randint(-6, 6))
    rands2.append(random.random()*random.randint(-6, 6))
    rands3.append(random.random()*random.randint(-6, 6))
    rands4.append(random.random()*random.randint(-6, 6))

new_data = []
for row in data:
    print sum(rands1)/10
    new_row = [
        row[0],
        row[1]-(sum(rands1)/10),
        row[2]-(sum(rands2)/10),
        row[3]-(sum(rands3)/10),
        row[4]-(sum(rands4)/10)
    ]
    if row[0] < 500:
        rands1 = rands1[1:] + [random.random()*random.randint(-6, 6)]
        rands2 = rands2[1:] + [random.random()*random.randint(-6, 6)]
        rands3 = rands3[1:] + [random.random()*random.randint(-6, 6)]
        rands4 = rands4[1:] + [random.random()*random.randint(-6, 6)]
    else:
        rands1 = rands1[1:] + [random.random()*4]
        rands2 = rands2[1:] + [random.random()*4]
        rands3 = rands3[1:] + [random.random()*4]
        rands4 = rands4[1:] + [random.random()*4]

    new_data.append(new_row)

col1 = [a[1] for a in new_data]
col2 = [a[2] for a in new_data]
col3 = [a[3] for a in new_data]
col4 = [a[4] for a in new_data]
ma_data = []
for i in range(len(col1[26:])):
    if (i < 120):
        ma_data.append([
            new_data[i+20][0],
            sum(col1[i-20:i+1]) / 21,
            sum(col2[i-20:i+1]) / 21,
            sum(col3[i-20:i+1]) / 21,
            sum(col4[i-20:i+1]) / 21
        ])
    else:
        ma_data.append([
            new_data[i + 15][0],
            sum(col1[i - 35:i + 1]) / 36,
            sum(col2[i - 35:i + 1]) / 36,
            sum(col3[i - 35:i + 1]) / 36,
            sum(col4[i - 35:i + 1]) / 36
        ])


nnew = np.array(ma_data)
df2 = pd.DataFrame(nnew)
df2.to_excel('file2.xlsx', index=False, header=False)