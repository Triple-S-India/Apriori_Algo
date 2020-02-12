#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[11]:


dataset = pd.read_csv("D:DS_TriS/Market_Basket_Optimisation.csv")


# In[12]:


dataset.head()


# In[18]:


dataset.shape


# In[19]:


transactions = []
for i in range(0, 7500):
    print(i)
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])


# In[22]:


# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)


# In[23]:


# Visualising the results
results = list(rules)


# In[ ]:




