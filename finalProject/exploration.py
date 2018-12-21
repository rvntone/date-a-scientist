import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from mpl_toolkits.mplot3d import Axes3D

#Create your df here:

df = pd.read_csv("profiles.csv")
#removing NAs values
df = df.dropna()

#Body Type column
print(df.body_type.head())
print(df.body_type.value_counts())
code = 0
body_type_mapping = {}
for body_type in df.body_type.value_counts().index:
    # print(body_type)
    body_type_mapping[body_type] = code
    code += 1
print(body_type_mapping)
df["body_type_code"] = df.body_type.map(body_type_mapping)

plt.hist(df.body_type_code)
plt.xlabel("body_type")
plt.ylabel("Frequency")
plt.xlim(-1, 13)
plt.show()


# print(df.smokes.head())
print(df.smokes.value_counts())
code = 0
smokes_mapping = {}
for smokes in df.smokes.value_counts().index:
    # print(smokes)
    smokes_mapping[smokes] = code
    code += 1
print(smokes_mapping)
df["smokes_code"] = df.smokes.map(smokes_mapping)

plt.hist(df.smokes_code)
plt.xlabel("Smokes")
plt.ylabel("Frequency")
plt.xlim(-1, 5)
plt.show()

# print(df.drinks.head())
print(df.drinks.value_counts())
code = 0
drinks_mapping = {}
for drinks in df.drinks.value_counts().index:
    # print(drinks)
    drinks_mapping[drinks] = code
    code += 1
# print(drinks_mapping)


df["drinks_code"] = df.drinks.map(drinks_mapping)
#
plt.hist(df.drinks_code)
plt.xlabel("Drinks")
plt.ylabel("Frequency")
plt.xlim(-1, 6)
plt.show()


# print(df.drugs.head())
print(df.drugs.value_counts())
code = 0
drugs_mapping = {}
for drugs in df.drugs.value_counts().index:
    # print(drugs)
    drugs_mapping[drugs] = code
    code += 1
# print(drugs_mapping)
df["drugs_code"] = df.drugs.map(drugs_mapping)
#
plt.hist(df.drugs_code)
plt.xlabel("Drugs")
plt.ylabel("Frequency")
plt.xlim(-1, 3)
plt.show()


# print(df.diet.head())
print(df.diet.value_counts())
code = 0
diet_mapping = {}
for diet in df.diet.value_counts().index:
    # print(diet)
    diet_mapping[diet] = code
    code += 1
# print(diet_mapping)
df["diet_code"] = df.diet.map(diet_mapping)
#
plt.hist(df.diet_code)
plt.xlabel("Diet")
plt.ylabel("Frequency")
plt.xlim(-1, 4)
plt.show()



# print(df.diet.head())
print(df.income.value_counts())
plt.hist(df.income)
plt.xlabel("Incomes")
plt.ylabel("Frequency")
plt.show()

# print(df.diet.head())
print(df.job.value_counts())
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

plt.hist(data_without_nonvalid_income.job_code)
plt.xlabel("Jobs")
plt.ylabel("Frequency")
plt.show()

# print(df.diet.head())
print(df.age.value_counts())
plt.hist(df.age)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()




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


