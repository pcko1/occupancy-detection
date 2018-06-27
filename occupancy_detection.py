from pandas import read_csv, DatetimeIndex
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import plot_model

# Evaluate the performance of a test
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

# Show the height value above each bar of a barchart
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%.3f' % height,
                ha='center', va='bottom')

# Define filenames
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

# Train a Random Forest
clf = RandomForestClassifier(n_estimators=10, max_depth=2, random_state=0)
clf = clf.fit(x_train,y_train)

# Show feature importance (only for Decision Trees / Random Forests)
importance = clf.feature_importances_

# Test the model by predicting the label of the test datasets
y_pred1_rf = clf.predict(x_test1)
y_pred2_rf = clf.predict(x_test2)

# Multi Layer Perceptron - Feedforward Artificial Neural Network
# Network Topology
def mlp_model():
    model = Sequential()
    model.add(Dense(50, input_dim=5, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.2)) 
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy']) 
    plot_model(model, to_file='Results/MLP_Model.png', show_shapes = True, show_layer_names = True)
    return model

# Add weights to classes
class_weights = class_weight.compute_class_weight('balanced', 
                                                  np.unique(y_train), 
                                                  y_train)
estimator = []
estimator.append(('standardize', StandardScaler()))
estimator.append(('mlp', KerasClassifier(build_fn=mlp_model, 
                                         epochs=2, 
                                         batch_size=50, 
                                         verbose=1, 
                                         class_weight=class_weights)))
# Pipeline of transforms and estimators
pipeline = Pipeline(estimator)

''' Takes time to run because it trains 10 different models
# Stratified k-fold validation to see how model scores on unseen data
seed = 7
np.random.seed(seed)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, x_train, y_train, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
'''

# Train the network
pipeline.fit(x_train, y_train)

# Test the network
y_pred1_mlp = pipeline.predict(x_test1)
y_pred2_mlp = pipeline.predict(x_test2)

# Prepare the data for One-Class SVM (semi-supervised)
x_train_svm = []
x_test_svm = []

for row in df_train.values:
    if row[-1] == 1: #occupied
        x_train_svm.append(row[0:5])
    else: #unoccupied
        x_test_svm.append(row[0:5])

x_test_svm = np.asarray(x_test_svm)
x_train_svm = np.asarray(x_train_svm)

# Train the OC-SVM on *OCCUPIED* data
ocsvm = svm.OneClassSVM(nu=0.01, kernel='poly', gamma=0.3, degree=2)
ocsvm.fit(x_train_svm)

# Test the OC-SVM
y_pred1_ocsvm = ocsvm.predict(x_test1)
y_pred2_ocsvm = ocsvm.predict(x_test2)

# Map OCSVM output from [-1,1] to [0,1] for compatibility
label_encoder = LabelEncoder()
label_encoder.fit([-1,1])
y_pred1_ocsvm = label_encoder.transform(y_pred1_ocsvm) 
y_pred2_ocsvm = label_encoder.transform(y_pred2_ocsvm)

## Results
# Evaluate all models on all test datasets
rf1 = Performance(y_test1, y_pred1_rf)
rf2 = Performance(y_test2, y_pred2_rf)
mlp1 = Performance(y_test1, y_pred1_mlp)
mlp2 = Performance(y_test2, y_pred2_mlp)
ocsvm1 = Performance(y_test1, y_pred1_ocsvm)
ocsvm2 = Performance(y_test2, y_pred2_ocsvm)

# Accuracy
fig, ax = plt.subplots(dpi=600)
ind = np.arange(2)
width = 0.15
acc_rf = (rf1.acc, rf2.acc)
bars_rf = ax.bar(ind, acc_rf, width, color='r')
acc_mlp = (mlp1.acc, mlp2.acc)
bars_mlp = ax.bar(ind + width, acc_mlp, width, color='y')
acc_ocsvm = (ocsvm1.acc, ocsvm2.acc)
bars_ocsvm = ax.bar(ind + 2*width, acc_ocsvm, width, color='g')
ax.set_ylim([0, 1.1])
ax.set_ylabel('Accuracy')
ax.set_title('Occupancy Detection Accuracy')
ax.set_xticks(ind + width)
ax.set_xticklabels(('Test 1', 'Test 2'))
ax.legend((bars_rf[0], bars_mlp[0], bars_ocsvm[0]), 
          ('RF', 'MLP', 'OCSVM'), loc='upper center')
autolabel(bars_rf)
autolabel(bars_mlp)
autolabel(bars_ocsvm)
plt.savefig('Results/Accuracy.png')

# F1 Score
fig, ax = plt.subplots(dpi=600)
ind = np.arange(2)
width = 0.15
f1_rf = (rf1.f1, rf2.f1)
bars_rf = ax.bar(ind, f1_rf, width, color='r')
f1_mlp = (mlp1.f1, mlp2.f1)
bars_mlp = ax.bar(ind + width, f1_mlp, width, color='y')
f1_ocsvm = (ocsvm1.f1, ocsvm2.f1)
bars_ocsvm = ax.bar(ind + 2*width, f1_ocsvm, width, color='g')
ax.set_ylim([0, 1.1])
ax.set_ylabel('F1 Score')
ax.set_title('F1 Score of Detectors')
ax.set_xticks(ind + width)
ax.set_xticklabels(('Test 1', 'Test 2'))
ax.legend((bars_rf[0], bars_mlp[0], bars_ocsvm[0]), 
          ('RF', 'MLP', 'OCSVM'), loc='upper center')
autolabel(bars_rf)
autolabel(bars_mlp)
autolabel(bars_ocsvm)
plt.savefig('Results/F1-Score.png')

# AUROC
plt.figure(figsize=(10,5), dpi=600)
lw = 2
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
# Test 1
plt.subplot(211)
plt.plot(rf1.fpr, rf1.tpr, color='red',
         lw=lw, label='RF - Test 1 (area = %0.2f)' % rf1.auroc)
plt.plot(mlp1.fpr, mlp1.tpr, color='green',
         lw=lw, label='MLP - Test 1 (area = %0.2f)' % mlp1.auroc)
plt.plot(ocsvm1.fpr, ocsvm1.tpr, color='blue',
         lw=lw, label='OCSVM - Test 1 (area = %0.2f)' % ocsvm1.auroc)
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
plt.annotate('Overlapping!', 
             xy=(0.014, 0.6), 
             arrowprops=dict(arrowstyle='->'), 
             xytext=(0.2, 0.7))
plt.title('Receiver Operating Characteristic')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
# Test 2
plt.subplot(212)
plt.plot(rf2.fpr, rf2.tpr, color='red',
         lw=lw, label='RF - Test 2 (area = %0.2f)' % rf2.auroc)
plt.plot(mlp2.fpr, mlp2.tpr, color='green',
         lw=lw, label='MLP - Test 2 (area = %0.2f)' % mlp2.auroc)
plt.plot(ocsvm2.fpr, ocsvm2.tpr, color='blue',
         lw=lw, label='OCSVM - Test 2 (area = %0.2f)' % ocsvm2.auroc)
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig('Results/AUROC.png')