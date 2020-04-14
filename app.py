import pickle
from urllib import request

from flask import Flask, render_template, request, session, url_for, redirect, logging
import pymysql
from flask_login import logout_user
import numpy as np

connection = pymysql.connect(host="localhost", user="root", password="", database="ticketfare")
cursor = connection.cursor()

app = Flask(__name__)
app.secret_key = 'random string'


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


# Dealing with the Missing Value

print("Original Length of Training Set : ", len(training_set))

training_set = training_set.dropna()

print("Length of Training Set after dropping NaN: ", len(training_set))


#Cleaning Journey Date 

#Training Set

training_set['Journey_Day'] = pd.to_datetime(training_set.Date_of_Journey, format='%d/%m/%Y').dt.day

training_set['Journey_Month'] = pd.to_datetime(training_set.Date_of_Journey, format='%d/%m/%Y').dt.month

# Test Set

test_set['Journey_Day'] = pd.to_datetime(test_set.Date_of_Journey, format='%d/%m/%Y').dt.day

test_set['Journey_Month'] = pd.to_datetime(test_set.Date_of_Journey, format='%d/%m/%Y').dt.month

# Compare the dates and delete the original date feature

training_set.drop(labels = 'Date_of_Journey', axis = 1, inplace = True)

test_set.drop(labels = 'Date_of_Journey', axis = 1, inplace = True)



# Cleaning Duration

# Training Set

duration = list(training_set['Duration'])

for i in range(len(duration)) :
    if len(duration[i].split()) != 2:
        if 'h' in duration[i] :
            duration[i] = duration[i].strip() + ' 0m'
        elif 'm' in duration[i] :
            duration[i] = '0h {}'.format(duration[i].strip())

dur_hours = []
dur_minutes = []  

for i in range(len(duration)) :
    dur_hours.append(int(duration[i].split()[0][:-1]))
    dur_minutes.append(int(duration[i].split()[1][:-1]))
    
training_set['Duration_hours'] = dur_hours
training_set['Duration_minutes'] =dur_minutes

training_set.drop(labels = 'Duration', axis = 1, inplace = True)


# Test Set

durationT = list(test_set['Duration'])

for i in range(len(durationT)) :
    if len(durationT[i].split()) != 2:
        if 'h' in durationT[i] :
            durationT[i] = durationT[i].strip() + ' 0m'
        elif 'm' in durationT[i] :
            durationT[i] = '0h {}'.format(durationT[i].strip())
            
dur_hours = []
dur_minutes = []  

for i in range(len(durationT)) :
    dur_hours.append(int(durationT[i].split()[0][:-1]))
    dur_minutes.append(int(durationT[i].split()[1][:-1]))
  
    
test_set['Duration_hours'] = dur_hours
test_set['Duration_minutes'] = dur_minutes

test_set.drop(labels = 'Duration', axis = 1, inplace = True)


#Cleaning Departure and Arrival Times

# Training Set


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

# Dependent Variable
Y_train = training_set.iloc[:,6].values  # 6 is the index of "Price" in the Training Set 

# Independent Variables
X_train = training_set.iloc[:,training_set.columns != 'Price'].values # selects all columns except "Price"

# Independent Variables for Test Set
X_test = test_set.iloc[:,:].values

Y_test = test_set.iloc[:,6].values 

from sklearn.preprocessing import LabelEncoder

le1 = LabelEncoder()
le2 = LabelEncoder()

# Training Set    

X_train[:,0] = le1.fit_transform(X_train[:,0])

X_train[:,1] = le1.fit_transform(X_train[:,1])

X_train[:,2] = le1.fit_transform(X_train[:,2])

X_train[:,3] = le1.fit_transform(X_train[:,3])

X_train[:,4] = le1.fit_transform(X_train[:,4])

X_train[:,5] = le1.fit_transform(X_train[:,5])

# Test Set


X_test[:,0] = le2.fit_transform(X_test[:,0])

X_test[:,1] = le2.fit_transform(X_test[:,1])

X_test[:,2] = le2.fit_transform(X_test[:,2])

X_test[:,3] = le2.fit_transform(X_test[:,3])

X_test[:,4] = le2.fit_transform(X_test[:,4])

X_test[:,5] = le2.fit_transform(X_test[:,5])



from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)

#sc_y = StandardScaler()

Y_train = Y_train.reshape((len(Y_train), 1)) 

Y_train = sc_X.fit_transform(Y_train)

Y_train = Y_train.ravel()






def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template("airfareprediction.html")


@app.route('/prediction', methods=["GET", "POST"])
def prediction():

        if request.method == "GET":
            age = request.form.get("Airline")
            gender = request.form.get("Date_of_Journey")
            cp = request.form.get("Source")

            Destination = request.form.get("Destination")

            trestbps = request.form.get("Route")

            chol = request.form.get("Dep_Time")
            fbs = request.form.get("Arrival_Time")
            restecg = request.form.get("Duration")
            thalach = request.form.get("Total_Stops")
            exang = request.form.get("Additional_Info")

            test_list = []

            #valofall = age + ',' + gender + ',' + cp + ',' + Destination + ',' + trestbps + ',' + chol + ',' + fbs + ',' + restecg + ',' + thalach + ',' + exang
            #valofall = '3'  + ',' + '2' + ',' + '5' + ',' + '20' + ',' + '4' + ',' + '8' + ',' + '2' + ',' + '3' + ',' + '2'+ ',' + '50' + ',' + '4' + ',' + '20' + ',' + '3' + ',' + '10'

            print(valofall)
            valofsplit = valofall.split(",")
            print(valofsplit)
            for i in range(0, len(valofsplit)):
                test_list.append((valofsplit[i]))
                # print(test_list)
            # X_std = scaler.fit_transform(X)
            print(test_list)

            from fileinput import filename
            filename = 'rf_mode.sav'
            loaded_model = pickle.load(open(filename, 'rb'))
            qqq =sc_X.inverse_transform(loaded_model.predict(np.array([test_list]))) #loaded_model.predict([test_list])
            print(qqq)
            # print("predicted heart disease is " + class_obt.get(y_gotdata[0]))

            print('kmeans')

            #rfresult = rf.predict([test_list])

            print('random forest')

            return render_template('PREDICTION.html',data=qqq[0])
        return render_template('prediction.html', user=session['user'])



@app.route('/register', methods=["POST", "GET"])
def register():
    if request.method == "POST":
        name = request.form.get("name")
        phone = request.form.get("phone")
        username = request.form.get("username")

        password = request.form.get("password")

        # print("insert into userdetails(fname, lname, password) values('"+fname+"','"+lname+",'"+password+")")

        # cursor.execute("insert into userdetails(fname, lname, password) values(:fname, :lname, :password)",{"fname":fname,"lname":fname,"password":password})
        cursor.execute(
            "insert into userdetails(name, phone, username ,password) values('" + name + "','" + phone + "','" + username + "','" + password + "')")

        connection.commit()

        return render_template("signin.html")
    else:
        return render_template("about.html")


@app.route('/login1', methods=["POST", "GET"])
def login1():
    if request.method == "POST":

        username = request.form.get("username")

        password = request.form.get("password")
        cursor.execute('SELECT * FROM userdetails WHERE username = %s AND password = %s', (username, password))
        # Fetch one record and return result
        account = cursor.fetchone()
        print(account)

        # If account exists in accounts table in out database
        if account:

            session['user'] = account[0]
            msg = 'Logged in successfully'

            return render_template('header.html', username=session['user'])
        else:
            # Account doesnt exist or username/password incorrect
            msg = 'Incorrect username/password!'
        # Show the login form with message (if any)
    return render_template('signin.html', msg=msg)


@app.route('/foo')
def foo():
    cursor.execute("select * from tweeterdata ORDER BY id DESC")
    data = cursor.fetchall()  # data from database
    msg = 'Incorrect username/password!'
    return render_template("alldata.html", value=data)


@app.route('/piechart')
def piechart():
    cursor.execute("select avg(psrate),avg(nsrate),avg(neutral) from tweeterdata")
    data = cursor.fetchall()  # data from database
    print(data)
    msg = 'Incorrect username/password!'
    return render_template("chart.html", value=data)


@app.route('/search1')
def search1():
    cursor.execute("select * from traindata  ")
    data = cursor.fetchall()  # data from database
    msg = 'Incorrect username/password!'
    return render_template("train.html", value=data)


@app.route('/search4')
def search4():
    cursor.execute("select * from test  ")
    data = cursor.fetchall()  # data from database
    msg = 'Incorrect username/password!'
    return render_template("test.html", value=data)


@app.route('/login')
def login():
    return render_template("signin.html")


@app.route('/register1')
def register1():
    return render_template("signup.html")


@app.route("/logout")
def logout():
    return render_template("index.html")


@app.route('/search12')
def search12():
    return render_template("search.html")

if __name__ == '__main__':
    app.run(debug="True")