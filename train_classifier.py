import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  #We will train our random forest classifier
from sklearn.metrics import accuracy_score

data_dict = pickle.load(open("./data.pickle", 'rb'))

# print(data_dict.keys())
# print(data_dict)

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)

print(f"{score * 100}% of samples were classified correctly!!!!!!")

with open('model.p', 'wb') as f:
    pickle.dump({'model' : model}, f)