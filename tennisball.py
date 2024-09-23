
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, precision_score, recall_score

# Feature Engineering (data assumed to be loaded)
tennis_ball_df_revised['Bounce Height Degradation'] = tennis_ball_df_revised['Bounce Height (cm)'] / tennis_ball_df_revised['Impacts']
tennis_ball_df_revised['Temp_Material_Interaction'] = tennis_ball_df_revised['Temperature (C)'] * tennis_ball_df_revised['Material'].factorize()[0]

# Define features and target
X = tennis_ball_df_revised[['Bounce Height (cm)', 'Bounce Speed (km/h)', 'Impacts', 'Temperature (C)', 
                            'Humidity (%)', 'Games Played', 'Bounce Height Degradation', 'Temp_Material_Interaction']]
y = np.where(tennis_ball_df_revised['Impacts'] > 1000, 0, 1)  # Binary classification for usable (1) or dead (0) balls

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
svm_model = SVC(kernel='linear', random_state=42)
nn_model = MLPClassifier(hidden_layer_sizes=(50, 30), max_iter=1000, random_state=42)

# Train models
rf_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)
nn_model.fit(X_train, y_train)

# Predictions
rf_predictions = rf_model.predict(X_test)
svm_predictions = svm_model.predict(X_test)
nn_predictions = nn_model.predict(X_test)

# Evaluation
results = {
    'Model': ['Random Forest', 'SVM', 'Neural Network'],
    'Accuracy': [accuracy_score(y_test, rf_predictions), 
                 accuracy_score(y_test, svm_predictions), 
                 accuracy_score(y_test, nn_predictions)],
    'MAE': [mean_absolute_error(y_test, rf_predictions), 
            mean_absolute_error(y_test, svm_predictions), 
            mean_absolute_error(y_test, nn_predictions)],
    'Precision': [precision_score(y_test, rf_predictions), 
                  precision_score(y_test, svm_predictions), 
                  precision_score(y_test, nn_predictions)],
    'Recall': [recall_score(y_test, rf_predictions), 
               recall_score(y_test, svm_predictions), 
               recall_score(y_test, nn_predictions)]
}

# Create results DataFrame
results_df = pd.DataFrame(results)
print(results_df)
