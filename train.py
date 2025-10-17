
import pandas as pd, joblib, json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# load data from folder
df = pd.read_csv("data/iris.csv")
print('Number of rows: ', df.shape[0])
X = df.iloc[:,:-1]; y = df.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=50, random_state=42)

# train model
model.fit(X_train, y_train)
preds = model.predict(X_test)
acc = float(accuracy_score(y_test, preds))
print('Accuracy score: ', acc)

# save model and metrics
joblib.dump(model, "models/model.joblib")

# write metrics both as CSV (human) and JSON (dvc metrics)
pd.DataFrame([{"accuracy":acc}]).to_csv("metrics/metrics.csv", index=False)
with open("metrics/metrics.json","w") as f:
    json.dump({"accuracy": acc}, f)

