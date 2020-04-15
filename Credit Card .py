#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


credit_df=pd.read_csv('D:\\creditcard.csv')
credit_df.head()


# In[4]:


credit_df.columns


# In[6]:


y=credit_df['Class'].values
y.shape


# In[7]:


x=credit_df[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']].values


# In[8]:


x.shape


# In[10]:


from sklearn.preprocessing import StandardScaler
x=StandardScaler().fit(x).transform(x)
x


# In[11]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.25,random_state=4)


# In[13]:


from sklearn.linear_model import LogisticRegression
fd=LogisticRegression(C=0.01,solver='liblinear').fit(x_train,y_train)


# In[14]:


y_hat=fd.predict(x_test)


# In[15]:


y_hat_probab=fd.predict_proba(x_test)


# In[21]:


from sklearn.metrics import jaccard_similarity_score,log_loss,classification_report
print(log_loss(y_test,y_hat))


# In[19]:


print(jaccard_similarity_score(y_test,y_hat))


# In[22]:


print(classification_report(y_test,y_hat))


# In[ ]:




