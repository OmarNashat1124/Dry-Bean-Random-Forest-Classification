from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

def load_dataset(path):
    return pd.read_csv(path)


def train_model(df, target_col="Class"):
    df.columns = df.columns.str.strip()
    for col in df.columns:
        if col != target_col:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    numeric_features = X.select_dtypes(include=["number"]).columns
    numeric_transformer = Pipeline(
        steps=[
            ("skew", PowerTransformer(method="yeo-johnson")),
            ("scale", StandardScaler())
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_features)],
        remainder="drop"
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
        ]
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\n Accuracy:", accuracy_score(y_test, y_pred))
    print("\n Classification Report:\n", classification_report(y_test, y_pred))
    return model
