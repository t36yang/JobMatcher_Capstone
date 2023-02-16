#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st


# In[2]:


import pickle


# In[3]:


from PIL import Image


# In[4]:


import requests


# In[5]:


import time


# In[6]:


import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
# preprocessing
import re
import string
import itertools # for flattening
# nltk library
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, regexp_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk import FreqDist
from wordcloud import WordCloud
from nltk.stem.wordnet import WordNetLemmatizer as wn
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.util import ngrams
nltk.download('punkt', quiet=True)
nltk.download('stopwords')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


# In[7]:


from nltk import WordNetLemmatizer # lemmatizer using WordNet
from nltk.corpus import wordnet # imports WordNet
from sklearn.feature_extraction.text import TfidfVectorizer


# In[8]:


import nltk
nltk.download('omw-1.4')


# In[9]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[10]:


st.title("Job Matcher")


# In[11]:


st.image('streamlit.jpeg')


# In[12]:


st.markdown("Enter all your skills that you can think of and separate with a comma! This system will return the jobs that align with your skillset.")


# In[13]:


#Add Sidebar
st.sidebar.markdown("## Match the jobs that align with your skillset")
st.sidebar.caption("About This System:")
st.sidebar.caption("This recommendation system was constructed by utilizing cosine similarities between your skillset and over 22,000 job descriptions that were trained using the NLTK NMF natural language processing algorithm..")


# In[14]:


# Sidebar cont.
st.sidebar.markdown("#### Informationa about the database:")
st.sidebar.caption("Downloaded from Kaggle: Dice Tech Job Board")
st.sidebar.caption("Please note that due to constraints on the data source, this system is only capable of matching jobs within the technology industry. Additionally, it should be noted that the dataset used was downloaded from Dice.com in 2018.")


# In[15]:


# Sidebar cont.
st.sidebar.markdown("#### Author: Vickie Yang")
st.sidebar.caption("Github: https://github.com/t36yang")
st.sidebar.caption("LinkedIn: www.linkedin.com/in/yangvickie")


# In[16]:


skill=st.text_input("Please enter your skills")


# In[17]:


userinput= skill


# In[18]:


new_df=pd.read_pickle('cleanedfile_dicejob.pickle')


# In[19]:


new_df[['City','State','Other']]=new_df.joblocation_address.str.split(", ", expand=True)


# In[20]:


new_df=new_df.drop('Other', axis=1)


# In[21]:


new_df.State=new_df.State.str.upper().apply(lambda x : "TX" if x == "TEXAS" else "DC" if x == "WASHINGTON" else "NY" if x == "SPRINGS" else x)


# In[47]:


st.write("---")
selectedState = st.selectbox('Select State:',
                    np.append(new_df.State.sort_values().unique(),["All"]))


# In[48]:


st.write("---")
'###### No. of Recommended Jobs to display'
jobs=st.slider('# of Recommended Jobs',0,100,1)
'You selected: ', jobs, 'jobs'


# In[74]:


def input_process(text):
    
    # get common stop words that we'll remove during tokenization/text normalization
    stop_words = stopwords.words('english')

    #initialize lemmatizer
    wnl = WordNetLemmatizer()

    # helper function to change nltk's part of speech tagging to a wordnet format.
    def pos_tagger(nltk_tag):
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:         
            return None
   

    # lower case everything
    txt_lower = text.lower()

    #remove mentions, hashtags, and urls, strip whitspace and breaks
    txt_lower = re.sub(r"@[a-z0-9_]+|#[a-z0-9_]+|http\S+", "", txt_lower).strip().replace("\r", "").replace("\n", "").replace("\t", "")
    
    #remove words with short length
    
    # remove stop words and punctuations 
    txt_norm = [x for x in word_tokenize(txt_lower) if ((x.isalpha()) & (x not in stop_words)) & (x not in ['good','great','found','company','lot','experience','fit','candidate','applicant','requirement','qualification','Deloitte','professional','year','application','opportunity','description','work','role','need','email','delivery',"req_id","job_req","req","id","please","resume","position","forward","receive","contact","minimum","required","disability","eligibility","employment","team","click"])]

    #  POS detection on the result will be important in telling Wordnet's lemmatizer how to lemmatize
    
    # creates list of tuples with tokens and POS tags in wordnet format
    txt_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tag(txt_norm))) 

    # lemmatize the input
    txt_processed = " ".join([wnl.lemmatize(x[0], x[1]) for x in txt_tagged if x[1] is not None])
    return txt_processed
    


# In[75]:


if st.button('Show Jobs'):
    
    with open('bestmodel.pkl' , 'rb') as f:
        lr = pickle.load(f)
        
    userresume=input_process(userinput)

    vectorized_col = new_df['joined_bigram']
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(vectorized_col)


    resume_vec = vectorizer.transform([userresume])

    resume_transform=lr.transform(resume_vec)
    sim = cosine_similarity(X_train, resume_vec)

    dic = {}
    for i,x in enumerate(sim):
      dic[i] = x

    highest_sim = pd.DataFrame(dic).T.sort_values(by = 0, ascending = False).head(jobs).index
    
    st.write("Jobs:")
    recommend=new_df.iloc[highest_sim, :]
    
    if selectedState=="All":
        st.dataframe(recommend[['jobtitle','jobdescription',"company",'advertiserurl']].rename(columns={"Job Title": 'jobtitle',"Job Description" : 'jobdescription',"Company": 'company','Link':'advertiserurl'}))
    else:
        st.dataframe(recommend.loc[recommend['State']==selectedState][['jobtitle','jobdescription',"company",'advertiserurl']].rename(columns={"Job Title": 'jobtitle',"Job Description" : 'jobdescription',"Company": 'company','Link':'advertiserurl'}))
    


# In[ ]:





# In[ ]:





# In[ ]:




