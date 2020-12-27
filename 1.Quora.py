#!/usr/bin/env python
# coding: utf-8

# ![Quora-1.png](attachment:Quora-1.png)

# <h1 style="text-align:center;font-size:30px;" > Quora Question Pairs </h1>

# <h1> 1. Business Problem </h1>

# <h2> 1.1 Description </h2>

# <p>Quora is a place to gain and share knowledge—about anything. It’s a platform to ask questions and connect with people who contribute unique insights and quality answers. This empowers people to learn from each other and to better understand the world.</p>
# <p>
# Over 100 million people visit Quora every month, so it's no surprise that many people ask similarly worded questions. Multiple questions with the same intent can cause seekers to spend more time finding the best answer to their question, and make writers feel they need to answer multiple versions of the same question. Quora values canonical questions because they provide a better experience to active seekers and writers, and offer more value to both of these groups in the long term.
# </p>
# <br>
# > Credits: Kaggle 
# 

# __ Problem Statement __
# - Identify which questions asked on Quora are duplicates of questions that have already been asked. 
# - This could be useful to instantly provide answers to questions that have already been answered. 
# - We are tasked with predicting whether a pair of questions are duplicates or not. 

# <h2> 1.2 Sources/Useful Links</h2>

# - Source : https://www.kaggle.com/c/quora-question-pairs
# <br><br>____ Useful Links ____
# - Discussions : https://www.kaggle.com/anokas/data-analysis-xgboost-starter-0-35460-lb/comments
# - Kaggle Winning Solution and other approaches: https://www.dropbox.com/sh/93968nfnrzh8bp5/AACZdtsApc1QSTQc7X0H3QZ5a?dl=0
# - Blog 1 : https://engineering.quora.com/Semantic-Question-Matching-with-Deep-Learning
# - Blog 2 : https://towardsdatascience.com/identifying-duplicate-questions-on-quora-top-12-on-kaggle-4c1cf93f1c30

# <h2>1.3 Real world/Business Objectives and Constraints </h2>

# 1. The cost of a mis-classification can be very high.
# 2. You would want a probability of a pair of questions to be duplicates so that you can choose any threshold of choice.
# 3. No strict latency concerns.
# 4. Interpretability is partially important.

# <h1>2. Machine Learning Probelm </h1>

# <h2> 2.1 Data </h2>

# <h3> 2.1.1 Data Overview </h3>

# <p> 
# - Data will be in a file Train.csv <br>
# - Train.csv contains 5 columns : qid1, qid2, question1, question2, is_duplicate <br>
# - Size of Train.csv - 60MB <br>
# - Number of rows in Train.csv = 404,290
# </p>

# <h3> 2.1.2 Example Data point </h3>

# <pre>
# "id","qid1","qid2","question1","question2","is_duplicate"
# "0","1","2","What is the step by step guide to invest in share market in india?","What is the step by step guide to invest in share market?","0"
# "1","3","4","What is the story of Kohinoor (Koh-i-Noor) Diamond?","What would happen if the Indian government stole the Kohinoor (Koh-i-Noor) diamond back?","0"
# "7","15","16","How can I be a good geologist?","What should I do to be a great geologist?","1"
# "11","23","24","How do I read and find my YouTube comments?","How can I see all my Youtube comments?","1"
# </pre>

# <h2> 2.2 Mapping the real world problem to an ML problem </h2>

# <h3> 2.2.1 Type of Machine Leaning Problem </h3>

# <p> It is a binary classification problem, for a given pair of questions we need to predict if they are duplicate or not. </p>

# <h3> 2.2.2 Performance Metric </h3>

# Source: https://www.kaggle.com/c/quora-question-pairs#evaluation
# 
# Metric(s): 
# * log-loss : https://www.kaggle.com/wiki/LogarithmicLoss
# * Binary Confusion Matrix

# <h2> 2.3 Train and Test Construction </h2>

# <p>  </p>
# <p> We build train and test by randomly splitting in the ratio of 70:30 or 80:20 whatever we choose as we have sufficient points to work with. </p>

# <h1>3. Exploratory Data Analysis </h1>

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from subprocess import check_output
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import os
import gc

import re
from nltk.corpus import stopwords
import distance
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup


# <h2> 3.1 Reading data and basic stats </h2>

# In[2]:


df = pd.read_csv("train.csv")

print("Number of data points:",df.shape[0])


# In[3]:


df.head()


# In[4]:


df.info()


# We are given a minimal number of data fields here, consisting of:
# 
# - id:  Looks like a simple rowID
# - qid{1, 2}:  The unique ID of each question in the pair
# - question{1, 2}:  The actual textual contents of the questions.
# - is_duplicate:  The label that we are trying to predict - whether the two questions are duplicates of each other.

# <h3> 3.2.1 Distribution of data points among output classes</h3>
# - Number of duplicate(smilar) and non-duplicate(non similar) questions

# In[5]:


df.groupby("is_duplicate")['id'].count().plot.bar()


# In[6]:


print('~> Total number of question pairs for training:\n   {}'.format(len(df)))


# In[7]:


print('~> Question pairs are not Similar (is_duplicate = 0):\n   {}%'.format(100 - round(df['is_duplicate'].mean()*100, 2)))
print('\n~> Question pairs are Similar (is_duplicate = 1):\n   {}%'.format(round(df['is_duplicate'].mean()*100, 2)))


# <h3> 3.2.2 Number of unique questions </h3>

# In[8]:


qids = pd.Series(df['qid1'].tolist() + df['qid2'].tolist())
unique_qs = len(np.unique(qids))
qs_morethan_onetime = np.sum(qids.value_counts() > 1)
print ('Total number of  Unique Questions are: {}\n'.format(unique_qs))
#print len(np.unique(qids))

print ('Number of unique questions that appear more than one time: {} ({}%)\n'.format(qs_morethan_onetime,qs_morethan_onetime/unique_qs*100))

print ('Max number of times a single question is repeated: {}\n'.format(max(qids.value_counts()))) 

q_vals=qids.value_counts()

q_vals=q_vals.values


# In[9]:


len(df['qid1'].tolist())
len(df['qid2'].tolist())
len(qids)
unique_qs
qids.value_counts()>1


# In[10]:



x = ["unique_questions" , "Repeated Questions"]
y =  [unique_qs , qs_morethan_onetime]

plt.figure(figsize=(10, 6))
plt.title ("Plot representing unique and repeated questions  ")
sns.barplot(x,y)
plt.show()


# <h3>3.2.3 Checking for Duplicates </h3>

# In[11]:


#checking whether there are any repeated pair of questions

pair_duplicates = df[['qid1','qid2','is_duplicate']].groupby(['qid1','qid2']).count().reset_index()

print ("Number of duplicate questions",(pair_duplicates).shape[0] - df.shape[0])


# <h3> 3.2.4 Number of occurrences of each question </h3>

# In[12]:


plt.figure(figsize=(20, 10))
plt.hist(qids.value_counts(), bins=160)
plt.yscale('log', nonposy='clip')
plt.title('Log-Histogram of question appearance counts')
plt.xlabel('Number of occurences of question')
plt.ylabel('Number of questions')
print ('Maximum number of times a single question is repeated: {}\n'.format(max(qids.value_counts()))) 


# <h3> 3.2.5 Checking for NULL values </h3>

# In[13]:


#Checking whether there are any rows with null values
nan_rows = df[df.isnull().any(1)]
print (nan_rows)


# - There are two rows with null values in question2 

# In[14]:


# Filling the null values with ' '
df = df.fillna('')
nan_rows = df[df.isnull().any(1)]
print (nan_rows)


# <h2>3.3 Basic Feature Extraction (before cleaning) </h2>

# Let us now construct a few features like:
#  - ____freq_qid1____ = Frequency of qid1's
#  - ____freq_qid2____ = Frequency of qid2's 
#  - ____q1len____ = Length of q1
#  - ____q2len____ = Length of q2
#  - ____q1_n_words____ = Number of words in Question 1
#  - ____q2_n_words____ = Number of words in Question 2
#  - ____word_Common____ = (Number of common unique words in Question 1 and Question 2)
#  - ____word_Total____ =(Total num of words in Question 1 + Total num of words in Question 2)
#  - ____word_share____ = (word_common)/(word_Total)
#  - ____freq_q1+freq_q2____ = sum total of frequency of qid1 and qid2 
#  - ____freq_q1-freq_q2____ = absolute difference of frequency of qid1 and qid2 

# In[15]:


if os.path.isfile('df_fe_without_preprocessing_train.csv'):
    df = pd.read_csv("df_fe_without_preprocessing_train.csv",encoding='latin-1')
else:
    df['freq_qid1'] = df.groupby('qid1')['qid1'].transform('count') 
    df['freq_qid2'] = df.groupby('qid2')['qid2'].transform('count')
    df['q1len'] = df['question1'].str.len() 
    df['q2len'] = df['question2'].str.len()
    df['q1_n_words'] = df['question1'].apply(lambda row: len(row.split(" ")))
    df['q2_n_words'] = df['question2'].apply(lambda row: len(row.split(" ")))

    def normalized_word_Common(row):
        w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
        w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
        return 1.0 * len(w1 & w2)
    df['word_Common'] = df.apply(normalized_word_Common, axis=1)

    def normalized_word_Total(row):
        w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
        w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
        return 1.0 * (len(w1) + len(w2))
    df['word_Total'] = df.apply(normalized_word_Total, axis=1)

    def normalized_word_share(row):
        w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
        w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
        return 1.0 * len(w1 & w2)/(len(w1) + len(w2))
    df['word_share'] = df.apply(normalized_word_share, axis=1)

    df['freq_q1+q2'] = df['freq_qid1']+df['freq_qid2']
    df['freq_q1-q2'] = abs(df['freq_qid1']-df['freq_qid2'])

    df.to_csv("df_fe_without_preprocessing_train.csv", index=False)

df.head()


# <h3> 3.3.1 Analysis of some of the extracted features </h3>

# - Here are some questions have only one single words.

# In[16]:


print ("Minimum length of the questions in question1 : " , min(df['q1_n_words']))

print ("Minimum length of the questions in question2 : " , min(df['q2_n_words']))

print ("Number of Questions with minimum length [question1] :", df[df['q1_n_words']== 1].shape[0])
print ("Number of Questions with minimum length [question2] :", df[df['q2_n_words']== 1].shape[0])


# <h4> 3.3.1.1 Feature: word_share </h4>

# In[17]:


plt.figure(figsize=(12, 8))

plt.subplot(1,2,1)
sns.violinplot(x = 'is_duplicate', y = 'word_share', data = df[0:])

plt.subplot(1,2,2)
sns.distplot(df[df['is_duplicate'] == 1.0]['word_share'][0:] , label = "1", color = 'red')
sns.distplot(df[df['is_duplicate'] == 0.0]['word_share'][0:] , label = "0" , color = 'blue' )
plt.show()


# - The distributions for normalized word_share have some overlap on the far right-hand side, i.e., there are quite a lot of questions with high word similarity
# - The average word share and Common no. of words of qid1 and qid2 is more when they are duplicate(Similar)

# <h4> 3.3.1.2 Feature: word_Common </h4>

# In[35]:


plt.figure(figsize=(12, 8))

plt.subplot(1,2,1)
sns.violinplot(x = 'is_duplicate', y = 'word_Common', data = df[0:])

plt.subplot(1,2,2)
sns.distplot(df[df['is_duplicate'] == 1.0]['word_Common'][0:] , label = "1", color = 'red')
sns.distplot(df[df['is_duplicate'] == 0.0]['word_Common'][0:] , label = "0" , color = 'blue' )
plt.show()


# <p> The distributions of the word_Common feature in similar and non-similar questions are highly overlapping </p>
