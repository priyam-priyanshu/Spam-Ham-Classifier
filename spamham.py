# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import re
import nltk
nltk.download('stopwords')


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

#ps = PorterStemmer()
lt = WordNetLemmatizer()



df = pd.read_csv("Dataset\SMSSpamCollection", sep="\t", names=["labels", "message"])

txt = []

for i in range(len(df)):
    words = re.sub("[^a-zA-Z]", " ", df["message"][i])
    words = words.lower()
    words = words.split()
    
    #words = [ps.stem(word) for word in words if word not in stopwords.words('english')]
    wrods = [lt.lemmatize(word) for word in words if word not in stopwords.words('english')]
    words = " ".join(words)
    txt.append(words)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)

x = cv.fit_transform(txt).toarray()

y=pd.get_dummies(df["labels"], drop_first=True)

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.20, random_state=0)

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()

model.fit(xtrain, ytrain)

pred = model.predict(xtest)