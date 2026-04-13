import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
 

df = pd.read_csv("mushrooms.csv")
print(f"Dataset shape : {df.shape}")
print(f"Class distribution:\n{df['class'].value_counts()}\n")


encoders = {}
df_encoded = df.copy()
for col in df.columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])
    encoders[col] = le


X = df_encoded.drop(columns=["class"])
y = df_encoded["class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
 
print(f"Training samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}\n")
 
results = {}
 
 
# Statystyki
def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=encoders["class"].classes_,
        zero_division=0
    )
    print(f"── {name} {'─' * (55 - len(name))}")
    print(f"Accuracy : {accuracy * 100:.2f}%")
    print(report)
    return accuracy
 
 
# 1. Random Forest
rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X_train, y_train)
results["Random Forest"] = evaluate("Random Forest", rf, X_test, y_test)
 
 
# 2. Logistic Regression
scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
 
lr = LogisticRegression(max_iter=100, random_state=42, solver="lbfgs")
lr.fit(X_train_s, y_train)
results["Logistic Regression"] = evaluate("Logistic Regression", lr, X_test_s, y_test)
 
 
# 3. KNN
knn = KNeighborsClassifier(n_neighbors=2, metric="euclidean")
knn.fit(X_train, y_train)
results["KNN"] = evaluate("KNN", knn, X_test, y_test)
 
 
# ── 5. Summary ────────────────────────────────────────────────────────────────
print("=" * 58)
print(f"{'PODSUMOWANIE DLA ACCURACY':^58}")
print("=" * 58)
print(f"  {'Algorytm':<35} {'Accuracy':>10}")
print("-" * 58)
for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"  {name:<35} {acc*100:>9.2f}%")
print("=" * 58)