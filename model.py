import pandas as pd

training_set = pd.read_excel("Data_Train.xlsx")
test_set = pd.read_excel("Test_set.xlsx")
# chechking the features in the Datasets

#Training Set

print("\nEDA on Training Set\n")
print("#"*30)

print("\nFeatures/Columns : \n", training_set.columns)
print("\n\nNumber of Features/Columns : ", len(training_set.columns))
print("\nNumber of Rows : ",len(training_set))
print("\n\nData Types :\n", training_set.dtypes)

print("\n Contains NaN/Empty cells : ", training_set.isnull().values.any())

print("\n Total empty cells by column :\n", training_set.isnull().sum(), "\n\n")


# Test Set
print("#"*30)
print("\nEDA on Test Set\n")
print("#"*30)


print("\nFeatures/Columns : \n",test_set.columns)
print("\n\nNumber of Features/Columns : ",len(test_set.columns))
print("\nNumber of Rows : ",len(test_set))
print("\n\nData Types :\n", test_set.dtypes)
print("\n Contains NaN/Empty cells : ", test_set.isnull().values.any())
print("\n Total empty cells by column :\n", test_set.isnull().sum())

print("Original Length of Training Set : ", len(training_set))

training_set = training_set.dropna()

print("Length of Training Set after dropping NaN: ", len(training_set))

training_set['Journey_Day'] = pd.to_datetime(training_set.Date_of_Journey, format='%d/%m/%Y').dt.day

training_set['Journey_Month'] = pd.to_datetime(training_set.Date_of_Journey, format='%d/%m/%Y').dt.month

# Test Set

test_set['Journey_Day'] = pd.to_datetime(test_set.Date_of_Journey, format='%d/%m/%Y').dt.day

test_set['Journey_Month'] = pd.to_datetime(test_set.Date_of_Journey, format='%d/%m/%Y').dt.month

# Compare the dates and delete the original date feature

training_set.drop(labels = 'Date_of_Journey', axis = 1, inplace = True)
print(training_set.drop)

test_set.drop(labels = 'Date_of_Journey', axis = 1, inplace = True)
duration = list(training_set['Duration'])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:
        if 'h' in duration[i]:
            duration[i] = duration[i].strip() + ' 0m'
        elif 'm' in duration[i]:
            duration[i] = '0h {}'.format(duration[i].strip())

dur_hours = []
dur_minutes = []

for i in range(len(duration)):
    dur_hours.append(int(duration[i].split()[0][:-1]))
    dur_minutes.append(int(duration[i].split()[1][:-1]))

training_set['Duration_hours'] = dur_hours
training_set['Duration_minutes'] = dur_minutes

training_set.drop(labels='Duration', axis=1, inplace=True)

# Test Set

durationT = list(test_set['Duration'])

for i in range(len(durationT)):
    if len(durationT[i].split()) != 2:
        if 'h' in durationT[i]:
            durationT[i] = durationT[i].strip() + ' 0m'
        elif 'm' in durationT[i]:
            durationT[i] = '0h {}'.format(durationT[i].strip())

dur_hours = []
dur_minutes = []

for i in range(len(durationT)):
    dur_hours.append(int(durationT[i].split()[0][:-1]))
    dur_minutes.append(int(durationT[i].split()[1][:-1]))

test_set['Duration_hours'] = dur_hours
test_set['Duration_minutes'] = dur_minutes

test_set.drop(labels='Duration', axis=1, inplace=True)
training_set['Depart_Time_Hour'] = pd.to_datetime(training_set.Dep_Time).dt.hour
training_set['Depart_Time_Minutes'] = pd.to_datetime(training_set.Dep_Time).dt.minute

training_set.drop(labels = 'Dep_Time', axis = 1, inplace = True)


training_set['Arr_Time_Hour'] = pd.to_datetime(training_set.Arrival_Time).dt.hour
training_set['Arr_Time_Minutes'] = pd.to_datetime(training_set.Arrival_Time).dt.minute

training_set.drop(labels = 'Arrival_Time', axis = 1, inplace = True)


# Test Set


test_set['Depart_Time_Hour'] = pd.to_datetime(test_set.Dep_Time).dt.hour
test_set['Depart_Time_Minutes'] = pd.to_datetime(test_set.Dep_Time).dt.minute


test_set.drop(labels = 'Dep_Time', axis = 1, inplace = True)

test_set['Arr_Time_Hour'] = pd.to_datetime(test_set.Arrival_Time).dt.hour
test_set['Arr_Time_Minutes'] = pd.to_datetime(test_set.Arrival_Time).dt.minute

test_set.drop(labels = 'Arrival_Time', axis = 1, inplace = True)
Y_train = training_set.iloc[:,6].values  # 6 is the index of "Price" in the Training Set

# Independent Variables
X_train = training_set.iloc[:,training_set.columns != 'Price'].values # selects all columns except "Price"

# Independent Variables for Test Set
X_test = test_set.iloc[:,:].values

Y_test = test_set.iloc[:,6].values
X_test
Y_test
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
svr = SVR(kernel = "rbf")

svr.fit(X_train,Y_train)

Y_pred = sc_X.inverse_transform(svr.predict(X_test))


pd.DataFrame(Y_pred, columns = ['Price']).to_excel("Final1_Pred.xlsx", index = False)
import pickle
from sklearn.ensemble import RandomForestRegressor
import numpy as np
filename = 'linear_svm_model1.sav'
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
pickle.dump(rf, open(filename, 'wb'))
rf.fit(X_train, Y_train);
predictions = rf.predict(X_test)
# Calculate the absolute errors
errors = abs(predictions - Y_test)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')