from pandas import read_csv, DatetimeIndex, DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_curve, auc
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

# Creae separate files for occupied / unoccupied conditions
occ_list, unocc_list = [], []

for (idx, row) in enumerate(df_train.values):
  if df_train.values[idx,-1]==0:
    unocc_list.append(df_train.values[idx,:])
  else:
    occ_list.append(df_train.values[idx,:])
occ_df, unocc_df = DataFrame(data=occ_list), DataFrame(data=unocc_list)
occ_df.to_csv('Datasets/OccupiedDataset.csv')
unocc_df.to_csv('Datasets/UnoccupiedDataset.csv')

# Split predictor and target variables
x_train = df_train.values[:,:-1]
y_train = df_train.values[:, -1]
x_test1 = df_test1.values[:,:-1]
y_test1 = df_test1.values[:, -1]
x_test2 = df_test2.values[:,:-1]
y_test2 = df_test2.values[:, -1]

# Train the tree
clf = RandomForestClassifier(n_estimators=10,max_depth=2, random_state=0)
clf = clf.fit(x_train,y_train)

# Show feature importance
importance = clf.feature_importances_

# Test the model by predicting the label of the test datasets
y_test1_pred = clf.predict(x_test1)
y_test2_pred = clf.predict(x_test2)

# Evaluate the model
# Calculate confusion matrix
cm1 = confusion_matrix(y_test1, y_test1_pred)
cm2 = confusion_matrix(y_test2, y_test2_pred)

# Calculate accuracy
acc1 = accuracy_score(y_test1, y_test1_pred)
acc2 = accuracy_score(y_test2, y_test2_pred)

# Calculate F1-score
f1_t1 = f1_score(y_test1, y_test1_pred)
f1_t2 = f1_score(y_test2, y_test2_pred)

# Calculate ROC
fpr1, tpr1, _ = roc_curve(y_test1, y_test1_pred)
fpr2, tpr2, _ = roc_curve(y_test2, y_test2_pred)

# Calculate AUROC
auroc1 = auc(fpr1,tpr1)
auroc2 = auc(fpr2,tpr2)
plt.figure()
lw = 2
plt.plot(fpr1, tpr1, color='green',
         lw=lw, label='Test 1 (area = %0.2f)' % auroc1)
plt.plot(fpr2, tpr2, color='red',
         lw=lw, label='Test 2 (area = %0.2f)' % auroc2)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()