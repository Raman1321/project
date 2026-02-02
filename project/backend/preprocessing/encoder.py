from sklearn.preprocessing import LabelEncoder, StandardScaler

def encode_and_scale(X):
    X = X.copy()
    encoders = {}

    categorical_cols = X.select_dtypes(include="object").columns
    numeric_cols = X.select_dtypes(exclude="object").columns

    # Encode categoricals
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    # Scale only numeric features
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    return X, encoders, scaler
