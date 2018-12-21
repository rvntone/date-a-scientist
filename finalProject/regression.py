import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from mpl_toolkits.mplot3d import Axes3D

#Create your df here:

df = pd.read_csv("profiles.csv")
df = df.dropna()
data_without_nonvalid_income = df[df.income != -1]
data_without_nonvalid_income = data_without_nonvalid_income[df.income < 100000]
# data_without_nonvalid_income = data_without_nonvalid_income[df.income > 20000]
data_without_nonvalid_income = data_without_nonvalid_income[df.age < 40]
# data_without_nonvalid_income = data_without_nonvalid_income[df.age > 20]
data_without_nonvalid_income = data_without_nonvalid_income[df.job != 'rather not say']
# data_without_nonvalid_income = data_without_nonvalid_income[df.job != 'unemployed']
data_without_nonvalid_income = data_without_nonvalid_income[df.job != 'other']
# data_without_nonvalid_income = data_without_nonvalid_income[df.job != 'retired']
# data_without_nonvalid_income = data_without_nonvalid_income[df.job != 'student']
print(data_without_nonvalid_income.job.value_counts())
print(data_without_nonvalid_income.income.value_counts())
print(data_without_nonvalid_income.age.value_counts())
print(data_without_nonvalid_income.groupby(['job']).mean().sort_values(by=['income']))


code = 0
job_mapping = {}
jobs = data_without_nonvalid_income.groupby(['job']).mean().sort_values(by=['income']).index
print(jobs);
for job in jobs:
    # print(body_type)
    job_mapping[job] = code
    code += 1
# print(job_mapping)
data_without_nonvalid_income["job_code"] = data_without_nonvalid_income.job.map(job_mapping)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# print(data_without_nonvalid_income["job_code"].values)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_without_nonvalid_income["job_code"].values, data_without_nonvalid_income["age"].values, data_without_nonvalid_income['income'].values,alpha=0.3);
plt.show();

values = []
job_codes = data_without_nonvalid_income["job_code"].values
incomes = data_without_nonvalid_income["income"].values
ages = data_without_nonvalid_income["age"].values

for i in zip(job_codes, ages):
    # values.append([i[0]])
    values.append([i[0], i[1]])
line_fitter = LinearRegression()

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(values, incomes, train_size=0.75, test_size=0.25, random_state=6)

line_fitter.fit(x_train, y_train)
incomes_predict = line_fitter.predict(x_test)
print("Train score:")
print(line_fitter.score(x_train, y_train))

print("Test score:")
print(line_fitter.score(x_test, y_test))

# ax.scatter(job_codes, ages, incomes, alpha=0.3);
# ax.plot([i[0] for i in x_test], [i[1] for i in x_test], incomes_predict, alpha=0.4, color='red');
# plt.show();
# print(line_fitter.score(values, ))


from sklearn.neighbors import KNeighborsRegressor



neighbors = np.arange(1, 200)
train_accuracy =np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i, k in enumerate(neighbors):
    regressor = KNeighborsRegressor(n_neighbors=k, weights="distance")
    regressor.fit(x_train, y_train)
    train_accuracy[i] = regressor.score(x_train, y_train)
    test_accuracy[i] = regressor.score(x_test, y_test)

plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')

plt.show();


# print(df.body_type.head())
# print(df.body_type.value_counts())
code = 0
body_type_mapping = {}
for body_type in df.body_type.value_counts().index:
    # print(body_type)
    body_type_mapping[body_type] = code
    code += 1
# print(body_type_mapping)
df["body_type_code"] = df.body_type.map(body_type_mapping)

# plt.hist(df.body_type_code)
# plt.xlabel("body_type")
# plt.ylabel("Frequency")
# plt.xlim(-1, 13)
# plt.show()


# print(df.smokes.head())
# print(df.smokes.value_counts())
code = 0
smokes_mapping = {}
for smokes in df.smokes.value_counts().index:
    # print(smokes)
    smokes_mapping[smokes] = code
    code += 1
# print(smokes_mapping)
df["smokes_code"] = df.smokes.map(smokes_mapping)

# plt.hist(df.smokes_code)
# plt.xlabel("Smokes")
# plt.ylabel("Frequency")
# plt.xlim(-1, 18)
# plt.show()

# print(df.drinks.head())
# print(df.drinks.value_counts())
code = 0
drinks_mapping = {}
for drinks in df.drinks.value_counts().index:
    # print(drinks)
    drinks_mapping[drinks] = code
    code += 1
# print(drinks_mapping)


df["drinks_code"] = df.drinks.map(drinks_mapping)
#
# plt.hist(df.drinks_code)
# plt.xlabel("Drinks")
# plt.ylabel("Frequency")
# plt.xlim(-1, 18)
# plt.show()


# print(df.drugs.head())
# print(df.drugs.value_counts())
code = 0
drugs_mapping = {}
for drugs in df.drugs.value_counts().index:
    # print(drugs)
    drugs_mapping[drugs] = code
    code += 1
# print(drugs_mapping)
df["drugs_code"] = df.drugs.map(drugs_mapping)
#
# plt.hist(df.drugs_code)
# plt.xlabel("Drugs")
# plt.ylabel("Frequency")
# plt.xlim(-1, 18)
# plt.show()


# print(df.diet.head())
# print(df.diet.value_counts())
code = 0
diet_mapping = {}
for diet in df.diet.value_counts().index:
    # print(diet)
    diet_mapping[diet] = code
    code += 1
# print(diet_mapping)
df["diet_code"] = df.diet.map(diet_mapping)
#
# plt.hist(df.drugs_code)
# plt.xlabel("Drugs")
# plt.ylabel("Frequency")
# plt.xlim(-1, 18)
# plt.show()

feature_data = df[['smokes_code', 'drinks_code', 'drugs_code', 'diet_code']]

x = feature_data.values
min_max_scaler = in_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)


feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)



colors = [
    '#1abc9c',
    '#2ecc71',
    '#3498db',
    '#9b59b6',
    '#34495e',
    '#f1c40f',
    '#e67e22',
    '#e74c3c',
    '#95a5a6',
    '#f39c12',
    '#c0392b',
    '#2980b9'
]
values = feature_data.values
labels = df.body_type_code.values
# print(df.body_type_code.value_counts())


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# print(values)
for i in range(len(df.body_type_code.value_counts())):
    points = np.array([values[j] for j in range(len(values)) if labels[j] == i])
    # ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors[i])

# plt.show()


from sklearn.model_selection import train_test_split


data_train, data_test, labels_train, labels_test = train_test_split(feature_data.values, labels,
                                                                    train_size=0.8, test_size=0.2, random_state=42)
# print(feature_data)

neighbors = np.arange(1, 200)
train_accuracy =np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# for i, k in enumerate(neighbors):
#     classifier = KNeighborsClassifier(n_neighbors=k)
#
#     # print(df.body_type)
#     classifier.fit(data_train, labels_train)
#
#     guesses = classifier.predict(data_test)
#
#     train_accuracy[i] = classifier.score(data_train, labels_train)
#
#     # Compute accuracy on the test set
#     test_accuracy[i] = classifier.score(data_test, labels_test)

# plt.title('KNeighbors Varying number of neighbors')
# plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
# plt.plot(neighbors, train_accuracy, label='Training accuracy')
# plt.legend()
# plt.xlabel('Number of neighbors')
# plt.ylabel('Accuracy')
# plt.show()

from sklearn.svm import SVC

gammasIncrementer = np.arange(0, 10)
gamma = 0

gammas = np.empty(len(gammasIncrementer))
accuracy = np.empty(len(gammasIncrementer))

# for i in gammasIncrementer:
#     gamma += 0.01
#     gammas[i] = gamma
#     classifier = SVC(kernel="poly", gamma=gamma, degree=5)
#     classifier.fit(data_train, labels_train)
#     accuracy[i] = classifier.score(data_test, labels_test)

# print(data_test[10])
# import random
#
# for x in range(10):
#     selected=random.randint(0, len(data_test))
#     print(classifier.predict([data_test[selected]]), labels_test[selected])
# plt.title('SVM Varying gamma value')
# plt.plot(gammas, accuracy, label='Accuracy')
# plt.legend()
# plt.xlabel('Number of neighbors')
# plt.ylabel('Accuracy')
# plt.show()

# print(df.job.values_count())

# x = []
# y = []
# print(df.body_type_code.values)
# body_type_code_values = df.body_type_code.values;
# diet_values = df.diet_code.values;
# print(body_type_code_values)
#
# plt.scatter(body_type_code_values, diet_values);
# plt.show();
# print(x)