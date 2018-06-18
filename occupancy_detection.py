from pandas import read_csv, DatetimeIndex
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_curve
from matplotlib import pyplot as plt

# Import data from CSV/txt file as Pandas DataFrame
def import_data(filename):
    dataset = read_csv(filename, header=0)
    # Drop timestamp (for now)
    dataset.drop('date', axis=1, inplace=True)
    # Convert datetime to time (remove date)
    #dataset['date'] = DatetimeIndex(dataset['date']).time
    # Check for NaN values
    if dataset.isnull().any().any():
        print('Dataset has NaN values, action required.')
    return dataset

train_filename = 'Datasets/datatraining.txt'
test1_filename = 'Datasets/datatest.txt'
test2_filename = 'Datasets/datatest2.txt'

# Import the UCI dataset
df_train = import_data(train_filename)
df_test1 = import_data(test1_filename)
df_test2 = import_data(test2_filename)

# Split predictor and target variables
x_train = df_train.values[:,:-1]
y_train = df_train.values[:, -1]
x_test1 = df_test1.values[:,:-1]
y_test1 = df_test1.values[:, -1]
x_test2 = df_test2.values[:,:-1]
y_test2 = df_test2.values[:, -1]

# Train the tree
#clf = tree.DecisionTreeClassifier()
clf = RandomForestClassifier(max_depth=10, random_state=0)
clf = clf.fit(x_train,y_train)

# Test the model by predicting the label of the test datasets
y_test1_pred = clf.predict(x_test1)

# Evaluate the model
# Calculate confusion matrix
cm = confusion_matrix(y_test1, y_test1_pred)

# Calculate accuracy
acc = accuracy_score(y_test1, y_test1_pred)

# Calculate F1-score
f1 = f1_score(y_test1, y_test1_pred)

# Calculate ROC
roc = roc_curve(y_test1, y_test1_pred)
