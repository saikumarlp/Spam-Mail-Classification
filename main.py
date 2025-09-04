import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle

email_data = pd.read_csv("spam.csv") # load the data in pandas
# print(email_data.head())

x = email_data.Message
y = email_data.Category

vectorizer = CountVectorizer()
x = vectorizer.fit_transform(x) 
model = MultinomialNB()

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.25,random_state=42)
model.fit(x_train,y_train)

# To check the model
msg = "Hello"
value = [msg]
vect = vectorizer.transform(value).toarray()
print(model.predict(vect))

# save the model
pickle.dump(model,open('spam.pkl','wb'))
pickle.dump(vectorizer,open('vec.pkl','wb'))

#To check the Model Accuracy
predictions = model.predict(x_test)
print(f"Model Accuracy : {round(accuracy_score(predictions,y_test),4)*100}%")
