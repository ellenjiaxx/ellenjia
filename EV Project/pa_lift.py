#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-import csv
# Python 3.8 is necessary
from nltk.corpus import stopwords
import re
import string
from sys import exit
import decimal
from collections import defaultdict
import pandas as pd
itr = 0
stop = stopwords.words('english')
df1 = pd.DataFrame(columns=['nb','nb2','value'])
inter1 = []
posts_all = []
posts_clean = []
keys_all = [] # array to store all keys from the ---- edmunds_pair_keys1.txt ---- file
file_length = 0 # calculate the number of rows in the file
results_dict = {'Pair':'Lift Value'};
dictionary1 = {}
d2_dict = defaultdict(dict)
# Define the input file name
filename = 'edmunds_new.csv'
# Define the textual column name
text_column = "Text"

try:
    df1_data = pd.read_csv(filename)
except:
    try:
        df1_data = pd.read_csv(filename, encoding = 'latin1')
    except:
        print("Your input file may have some format issues. Please contact TA for more details")

df1_data = df1_data.dropna(subset=[text_column])
inter1 = df1_data[text_column].to_list()
file_length = len(df1_data) + 1

for row in inter1:
    out1 = re.sub('[%s]' % re.escape(string.punctuation), '', row.lower())
    posts_all.append(out1)

for post in posts_all:
    s = []
    for i in post.split():
        if i not in stop:
            s.append(i)
    posts_clean.append(s)


print ("WARNING::EVERYTHING NEEDS TO BE IN LOWER CASE in edmunds_pair_keys.txt file")
with open('edmunds_pair_keys.txt') as fileText: # Make sure this file does not contain any empty lines.
    for row in fileText:
        keys_all = row.strip().split(",")
    length = len(keys_all)
    for index in range(len(keys_all)):
        subs_counter = index + 1
        while subs_counter < len(keys_all):

            nb = keys_all[index]
            nb2 = keys_all[subs_counter]
            print ('-------------------' + nb + ' and ' + nb2 + '-------------------')
            subs_counter = subs_counter + 1

            nb_plu = nb + 's'
            nb_app = nb + "'s"

            nb2_plu = nb2 + 's'
            nb2_app = nb2 + "'s"

            for post in posts_clean:
                for n,word in enumerate(post):
                    if(word == nb_plu or word == nb_app):
                        post[n] = nb
                    elif(word == nb2_plu or word == nb2_app):
                        post[n] = nb2

            for post in posts_clean:
                for word in post:
                    dictionary1[word] = 0

            for post in posts_clean:
                x = {}
                for word in post:
                    x[word] = 1
                for word in post:
                    if (x[word] > 0):
                        dictionary1[word] = dictionary1[word] + 1
                    x[word] = -1

            for post in posts_clean:
                for word in post:
                    for word2 in post:
                        if(word != word2):
                            d2_dict[word][word2] = 0

            for post in posts_clean:
                d3_dict = defaultdict(dict)
                for word in post:
                    for word2 in post:
                        if(word != word2):
                            d3_dict[word][word2] = 1

                for word in post:
                    for word2 in post:
                        if(word != word2):
                            if (d3_dict[word][word2] > 0):
                                d2_dict[word][word2] = d2_dict[word][word2] + 1
                            d3_dict[word][word2] = -1


            if(nb in dictionary1):
                print(nb,end=" ")
                print(dictionary1[nb])
                #print (nb + " " + str(dictionary1[nb]))
            else:
                print ("This word is not present")
                exit()

            if(nb2 in dictionary1):
                print(nb2,end=" ")
                print(dictionary1[nb2])
                #print (nb2 + " " + str(dictionary1[nb2]))
            else:
                print (nb2, "This word is not present")
                exit()

            if(nb in d2_dict):
                if(nb2 in d2_dict[nb]):
                    print(nb +" " +nb2,end=" ")
                    print(d2_dict[nb][nb2])
                    #print (nb +" " +nb2 +" " + str(d2_dict[nb][nb2]))
                    results_dict.update({nb + "_" +nb2:decimal.Decimal(decimal.Decimal(file_length*(d2_dict[nb][nb2]))/decimal.Decimal((dictionary1[nb]*dictionary1[nb2])))})
                    print ('lift('+ nb + "," +nb2 + ')',decimal.Decimal(decimal.Decimal(file_length*(d2_dict[nb][nb2]))/decimal.Decimal((dictionary1[nb]*dictionary1[nb2]))))
                    val = decimal.Decimal(decimal.Decimal(file_length*(d2_dict[nb][nb2]))/decimal.Decimal((dictionary1[nb]*dictionary1[nb2])))
                    df1.loc[itr] = [nb, nb2, val]  # adding a row
                    itr = itr + 1
                else:
                    print ("These words are not present together in a post")
                    exit()
            else:
                print ("These words are not present together in a post")
                exit()
            df1.to_csv('Lift_Values.csv')
print ('--------------------------------------------------------\n')


df2 = df1.copy()
df2 = df2.rename(columns={'nb': 'k', 'nb2': 'nb'})
df2 = df2.rename(columns={'k': 'nb2'})
frames = [df1, df2]
result = pd.concat(frames, sort=True)
result = result.reset_index(drop=True)
result['value'] = result['value'].astype(float)
table = pd.pivot_table(result, values='value', index='nb',columns='nb2', fill_value=0.00)
print ('----------------------------Lift Table (full)----------------------------')
print (table)
dis_table = pd.pivot_table(result, values='value', index='nb',columns='nb2', fill_value=0.00)


for i in range(table.shape[0] ):
    for j in range(table.shape[1]):
        if i>=j:
            table.iloc[i,j] = " "
            dis_table.iloc[i,j] = " "
        else:
            dis_table.iloc[i,j] = 1/dis_table.iloc[i,j]
print ('----------------------------Lift Table----------------------------')
print(table)
print ('----------------------------Dissimilarity Matrix----------------------------')
print(dis_table)
table.to_csv('Lift_Matrix.csv')
dis_table.to_csv("dissimilarity_matrix.csv")
print ('Consolidated lift values and matrix can be found in Lift_Values.csv file and Lift_Matrix.csv')
print ('Dissimilarity Matrix can be found in dissimilarity_matrix.csv')
