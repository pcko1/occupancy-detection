from pandas import read_csv, DatetimeIndex, DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_curve, auc
from sklearn.model_selection import cross_val_score, StratifiedKFold
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.metrics import binary_accuracy
from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np


class Performance:
    def __init__(self, y_test, y_pred):
        self.cm = confusion_matrix(y_test, y_pred)
        self.acc = accuracy_score(y_test, y_pred)
        self.f1 = f1_score(y_test, y_pred)
        self.fpr, self.tpr, _ = roc_curve(y_test, y_pred)
        self.auroc = auc(self.fpr,self.tpr)
             
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

## Train a Random Forest
clf = RandomForestClassifier(n_estimators=10,max_depth=2, random_state=0)
clf = clf.fit(x_train,y_train)

# Show feature importance (only for Decision Trees / Random Forests)
importance = clf.feature_importances_

# Test the model by predicting the label of the test datasets
y_pred1_rf = clf.predict(x_test1)
y_pred2_rf = clf.predict(x_test2)

## Multi Layer Perceptron - Feedforward Artificial Neural Network
# Build and train network
def mlp_model():
    model = Sequential()
    model.add(Dense(100, input_dim=5, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])    
    return model
seed = 7
np.random.seed(seed)
estimator = KerasClassifier(build_fn=mlp_model, epochs=10, batch_size=200, verbose=1)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, x_train, y_train, cv=kfold)

print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
'''
history = model.fit(x_train, y_train, epochs=10, batch_size=200, validation_data=(x_test1, y_test1), verbose=1, shuffle=False)
# Test the network
y_pred1_mlp = model.predict(x_test1)
y_pred2_mlp = model.predict(x_test2)


# Evaluate the model
rf1 = Performance(y_test1, y_pred1_rf)
rf2 = Performance(y_test2, y_pred2_rf)
mlp1 = Performance(y_test1, y_pred1_mlp)
mlp2 = Performance(y_test1, y_pred2_mlp)

plt.figure()
lw = 2
plt.plot(rf1.fpr, rf1.tpr, color='green',
         lw=lw, label='Test 1 (area = %0.2f)' % rf1.auroc)
plt.plot(mlp1.fpr, mlp1.tpr, color='red',
         lw=lw, label='mlp Test 2 (area = %0.2f)' % mlp1.auroc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
'''