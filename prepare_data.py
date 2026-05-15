import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

def load_and_prepare_data(filepath='composite_data.csv'):
    """Загружает, чистит данные и возвращает X, y_mod, y_strength, y_ratio"""
    df = pd.read_csv(filepath)

    def remove_outliers_iqr(df, columns):
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower) & (df[col] <= upper)]
        return df

    feature_cols = ['Соотношение матрица-наполнитель', 'Плотность, кг/м3',
                    'модуль упругости, ГПа', 'Количество отвердителя, м.%',
                    'Содержание эпоксидных групп,%_2', 'Температура вспышки, С_2',
                    'Поверхностная плотность, г/м2', 'Потребление смолы, г/м2',
                    'Угол нашивки, град', 'Шаг нашивки', 'Плотность нашивки']

    target_modulus = 'Модуль упругости при растяжении, ГПа'
    target_strength = 'Прочность при растяжении, МПа'
    target_ratio = 'Соотношение матрица-наполнитель'

    df_clean = remove_outliers_iqr(df, feature_cols)

    X = df_clean[feature_cols].copy()
    y_mod = df_clean[target_modulus]
    y_strength = df_clean[target_strength]
    y_ratio = df_clean[target_ratio]

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

    # Исправленный перенос строки (без обратных слешей)
    (X_train, X_test, y_mod_train, y_mod_test,
     y_strength_train, y_strength_test,
     y_ratio_train, y_ratio_test) = train_test_split(
         X_scaled, y_mod, y_strength, y_ratio, test_size=0.2, random_state=42)

    return (X_train, X_test, y_mod_train, y_mod_test,
            y_strength_train, y_strength_test,
            y_ratio_train, y_ratio_test, scaler)
