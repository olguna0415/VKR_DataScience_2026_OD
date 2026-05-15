import joblib
from sklearn.ensemble import GradientBoostingRegressor
from prepare_data import load_and_prepare_data

def train_models():
    (X_train, X_test, y_mod_train, y_mod_test,
     y_strength_train, y_strength_test, _, _, _) = load_and_prepare_data()

    gb_modulus = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    gb_modulus.fit(X_train, y_mod_train)

    gb_strength = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    gb_strength.fit(X_train, y_strength_train)

    joblib.dump(gb_modulus, 'model_modulus.pkl')
    joblib.dump(gb_strength, 'model_strength.pkl')

    print("✅ Модели машинного обучения обучены и сохранены.")
    return gb_modulus, gb_strength

if __name__ == '__main__':
    train_models()
