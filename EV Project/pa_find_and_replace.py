#!/usr/bin/env python3.8

import pandas as pd
import re
# Define the input file name
filename = 'edmunds_extraction.csv'
# Define the output file name
output = 'edmunds_new.csv'
# Define the textual column name
text_column = "Text"
# The list used to stored the replaced csv files
output_list=[]

try:
    df1 = pd.read_csv(filename)
except:
    try:
        df1 = pd.read_csv(filename, encoding = 'latin1')
    except:
        print("Your input file may have some format issues. Please contact TA for more details")

df2 = pd.read_csv('models.csv',header=None)
df2 = df2.dropna(subset=[0])
df2 = df2.dropna(subset=[1])
replace_list = df2.values.tolist()

try:
    df1 = df1.dropna(subset=[text_column])
    text = df1[text_column]
except:
    print("Please make sure the column '{}' is in the input file".format(text_column))

replaced_text = []
for x in text:
    x = re.sub('[%s]' % re.escape("'"), '', x.lower())
    for dic in replace_list:
        x = x.lower().replace(dic[1].lower(),"" + dic[0].lower() + "")
    replaced_text.append(x)

df1[text_column] = replaced_text
df1.to_csv(output,index=False)
print("Wrote to edmunds_new.csv")
