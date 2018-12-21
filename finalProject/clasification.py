import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import random

#Create your df here:

df = pd.read_csv("profiles.csv")
df = df.dropna()


code = 0
body_type_mapping = {}
for body_type in df.body_type.value_counts().index:
    body_type_mapping[body_type] = code
    code += 1
df["body_type_code"] = df.body_type.map(body_type_mapping)

code = 0
smokes_mapping = {}
for smokes in df.smokes.value_counts().index:
    smokes_mapping[smokes] = code
    code += 1
df["smokes_code"] = df.smokes.map(smokes_mapping)

code = 0
drinks_mapping = {}
for drinks in df.drinks.value_counts().index:
    drinks_mapping[drinks] = code
    code += 1
df["drinks_code"] = df.drinks.map(drinks_mapping)

code = 0
drugs_mapping = {}
for drugs in df.drugs.value_counts().index:
    drugs_mapping[drugs] = code
    code += 1
df["drugs_code"] = df.drugs.map(drugs_mapping)


code = 0
diet_mapping = {}
for diet in df.diet.value_counts().index:
    diet_mapping[diet] = code
    code += 1
df["diet_code"] = df.diet.map(diet_mapping)


feature_data = df[['smokes_code', 'drinks_code', 'drugs_code', 'diet_code']]

x = feature_data.values
min_max_scaler = in_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)


feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)
values = feature_data.values
labels = df.body_type_code.values

data_train, data_test, labels_train, labels_test = train_test_split(feature_data.values, labels,
                                                                    train_size=0.8, test_size=0.2, random_state=42)
neighbors = np.arange(1, 150)
train_accuracy =np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i, k in enumerate(neighbors):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(data_train, labels_train)
    guesses = classifier.predict(data_test)
    train_accuracy[i] = classifier.score(data_train, labels_train)
    test_accuracy[i] = classifier.score(data_test, labels_test)

plt.title('KNeighbors Varying number of neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()


gammasIncrementer = np.arange(0, 100)
gamma = 0

gammas = np.empty(len(gammasIncrementer))
accuracy = np.empty(len(gammasIncrementer))

for i in gammasIncrementer:
    gamma += 0.1
    gammas[i] = gamma
    classifier = SVC(kernel="rbf", gamma=gamma)
    classifier.fit(data_train, labels_train)
    accuracy[i] = classifier.score(data_test, labels_test)

for x in range(10):
    selected=random.randint(0, len(data_test))
    print(classifier.predict([data_test[selected]]), labels_test[selected])
plt.title('SVM Varying gamma value')
plt.plot(gammas, accuracy, label='Accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()

print(df.job.values_count())

x = []
y = []
print(df.body_type_code.values)
body_type_code_values = df.body_type_code.values;
diet_values = df.diet_code.values;
print(body_type_code_values)

plt.scatter(body_type_code_values, diet_values);
plt.show();
print(x)