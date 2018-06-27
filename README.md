# occupancy-detection
My attempt on the UCI Occupancy Detection dataset (https://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+) using various methods.

Scoring >98% with a Random Forest as of now.

## Performance metrics:

### Test 1 - 2665 Samples

Accuracy: 0.98

AUC-ROC : 0.98

F1-score: 0.97

### Test 2 - 9752 Samples

Accuracy: 0.99

AUC-ROC : 0.99

F1-score: 0.98

## Dataset Information:
Three datasets are available, of which one is for training (8143 samples) and two are for testing (2665+9752 samples). There are 7 attributes in total, including the occupancy label.

The timestamp is dropped because the algorithm already achieves very high performance, so only minor improvements are expected by incorporating it into the model.

## Attribute Information:
1. **Light** - Lux
2. **CO2** - ppm
3. **Temperature** - Celsius
4. **Humidity Ratio** - No unit. Derived quantity from temperature and relative humidity, in kgwater-vapor/kg-air 
5. **Relative Humidity** - %

## Attribute Importance (based on Gini Impurity of the Random Forest):
1. Light - 0.55
2. CO2 - 0.23
3. Temperature - 0.18
4. Humidity Ratio - 0.03  (negligible)
5. Relative Humidity - 0.001 (negligible)

## Results
![Accuracy](Results/Accuracy.png)
![F1-Score](Results/F1-Score.png)
![AUROC](Results/AUROC.png)

## Run
```python3 occupancy_detection.py```

## Dependencies
This code has been developed with the following packages:
* matplotlib v2.2.2
* sklearn v0.19.1
* pandas v0.22.0
* keras v2.1.6
* tensorflow v1.7.0
