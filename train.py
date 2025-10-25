import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline

def train_model():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score

    # === 1. Wczytanie danych ===
    url = "adresowo_lodz_cleaned.csv"
    df = pd.read_csv(url).dropna(subset=['price_total_zl_cleaned', 'area_m2'])

    # === 2. Definicja cech i celu ===
    X = df[['area_m2', 'locality', 'rooms', 'owner_direct', 'photos', 'date_posted']]
    y = df['price_total_zl_cleaned']

    # === 3. Podział na zbiory treningowy/testowy ===
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # === 4. Imputacja (opcjonalna) ===
    imputer = IterativeImputer(random_state=0)

    # === 5. Definicja kolumn numerycznych i kategorycznych ===
    numeric_features = ['area_m2', 'rooms', 'photos']
    categorical_features = ['locality', 'owner_direct', 'date_posted']

    # === 6. ColumnTransformer: preprocessing ===
    preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', imputer),
            #('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
    )
    # === 6. ColumnTransformer: preprocessing ===


    # === 7. Pipeline z modelem ===
    pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', DecisionTreeRegressor(
        max_depth=4,
        random_state=42,
    ))
    ])

    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score
    # === 8. Trening ===
    pipeline.fit(X_train, y_train)

    # === 9. Predykcja i ocena ===
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"R²: {r2:.3f}")
    # === 12. Zapis modelu ===
    joblib.dump(pipeline, "model_random_forest_adresowo_lodz.pkl")
    print("✅ Model zapisano jako 'model_random_forest_adresowo_lodz.pkl'")

    return r2