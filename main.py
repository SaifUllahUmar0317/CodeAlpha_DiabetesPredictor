import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
import joblib
from sklearn.svm import SVC

# Data loading
data = pd.read_csv("diabetes.csv")
data.head()

# EDA
data.duplicated().sum()
data.isnull().sum()
data.Outcome.value_counts()

# Feature Selection
sns.heatmap(data.corr(),annot = True, cmap = "Blues")
for i in range(8):
    cor = data.iloc[:,i].corr(data.iloc[:,8])
    print(f"Correlation b/w {data.columns[i]} and {data.columns[8]} = {cor: .4f}")
for col in data.columns:
    if (col != "Outcome"):
        info = mutual_info_classif(data[[col]], data["Outcome"])
        print(f"MI b/w {col} and Outcome = {info[0]:.4f}")
X = data.drop("Outcome", axis = 1)
Y = data["Outcome"]

# train test splitting
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y, test_size=0.2, random_state=42)

# Scalling
scaler = MinMaxScaler()
Xtrain_scaled = scaler.fit_transform(Xtrain)
Xtest_scaled = scaler.transform(Xtest)

# Balancing the dataset
smote = SMOTE(random_state = 42)
Xtrain_resampled, Ytrain_resampled = smote.fit_resample(Xtrain_scaled, Ytrain)

# Model selection and hyperparameters tunning with GridSearchCV
models = {
    "DecisionTree": {
        "model": DecisionTreeClassifier(), 
        "params":{
            "criterion": ["gini", "entropy"],
            "max_depth": [5,6,7,8,9,10],
            "min_samples_split": [2,4,6,8,10]
            }
    },
    
    "KNN": {
        "model": KNeighborsClassifier(),
        "params": { 
            "n_neighbors": [2,4,6,8,10] 
        }
    },
    
    "RandomForest": {
        "model": RandomForestClassifier(),
        "params": {
            "n_estimators": [100,150,200],
            "max_depth": [5,7,9,10],
            "min_samples_split": [2,4,6,8]
        }
    },
    
    "SVC":{
        "model": SVC(),
        "params": {
            "C": [0.1,1,10,100],
            "gamma": [0.1,1,10]
        }
    }
}

scores = []
final_model = None
best_score = 0
for name,model_params in models.items():
    best_model_params = GridSearchCV(model_params["model"], param_grid=model_params["params"], cv=5)
    best_model_params.fit(Xtrain_resampled, Ytrain_resampled)
    
    if best_model_params.best_score_ > best_score:
        best_score = best_model_params.best_score_
        final_model = best_model_params.best_estimator_
   
    scores.append({
        "model": name,
        "best parameters": best_model_params.best_params_,
        "score": best_model_params.best_score_
    })

params_scores_df = pd.DataFrame(scores)
print(params_scores_df)

# Prediction and Evaluating the model
Ypred = final_model.predict(Xtest_scaled)

cm = confusion_matrix(Ytest, Ypred)
acc = accuracy_score(Ytest, Ypred)
pre = accuracy_score(Ytest, Ypred)
recall = accuracy_score(Ytest, Ypred)
f1_score = accuracy_score(Ytest, Ypred)
print(f"Accuracy score: {acc:.2f}")
print(f"Precision score: {pre:.2f}")
print(f"Recall score: {recall:.2f}")
print(f"F1 score: {f1_score:.2f}")

sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["No diabetes", "diabetes"],
           yticklabels=["No diabetes", "diabetes"])
plt.title("Confusion Matrix")
plt.show()

# Saving the model and scaler
with open("Diabetes pedictor model.pkl", 'wb') as file:
    joblib.dump(final_model, file)
with open("Scaler.pkl", 'wb') as f:
    joblib.dump(scaler, f)
