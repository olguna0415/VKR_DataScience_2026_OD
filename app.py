import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

try:
    model_modulus = joblib.load('model_modulus.pkl')
    model_strength = joblib.load('model_strength.pkl')
    model_ratio = load_model('ratio_nn.h5')
    scaler = joblib.load('scaler.pkl')
    print("✅ Все модели загружены.\n")
except Exception as e:
    print("❌ Ошибка загрузки моделей. Убедитесь, что файлы model_modulus.pkl, model_strength.pkl, ratio_nn.h5, scaler.pkl находятся в папке.")
    exit(1)

FEATURES = [
    'Соотношение матрица-наполнитель',
    'Плотность, кг/м3',
    'модуль упругости, ГПа',
    'Количество отвердителя, м.%',
    'Содержание эпоксидных групп,%_2',
    'Температура вспышки, С_2',
    'Поверхностная плотность, г/м2',
    'Потребление смолы, г/м2',
    'Угол нашивки, град',
    'Шаг нашивки',
    'Плотность нашивки'
]

def get_user_input():
    print("Введите характеристики композита (можно копировать строку с числами через пробел):")
    print("Порядок признаков:")
    for i, f in enumerate(FEATURES, 1):
        print(f"  {i}. {f}")

    while True:
        try:
            values = input("\nВведите 11 чисел через пробел: ").strip().split()
            if len(values) != 11:
                print(f"Ошибка: нужно ровно 11 значений. Вы ввели {len(values)}.")
                continue
            values = [float(v) for v in values]
            return np.array(values).reshape(1, -1)
        except ValueError:
            print("Ошибка: введите только числа, разделённые пробелами.")

def predict_modulus_strength(X_raw):
    X_scaled = scaler.transform(X_raw)
    mod_pred = model_modulus.predict(X_scaled)[0]
    str_pred = model_strength.predict(X_scaled)[0]
    return mod_pred, str_pred

def predict_ratio(X_raw):
    X_scaled = scaler.transform(X_raw)
    ratio_pred = model_ratio.predict(X_scaled, verbose=0)[0, 0]
    return ratio_pred

def main():
    print("\n" + "="*50)
    print("ПРОГНОЗИРОВАНИЕ СВОЙСТВ КОМПОЗИТНЫХ МАТЕРИАЛОВ")
    print("="*50)

    X_raw = get_user_input()

    print("\nВыберите тип прогноза:")
    print("1 - Только модуль упругости и прочность при растяжении")
    print("2 - Только соотношение матрица-наполнитель")
    print("3 - Оба прогноза")
    choice = input("Ваш выбор (1/2/3): ").strip()

    if choice == '1':
        mod, st = predict_modulus_strength(X_raw)
        print(f"\n📊 Результаты (модели машинного обучения):")
        print(f"   Модуль упругости при растяжении: {mod:.2f} ГПа")
        print(f"   Прочность при растяжении: {st:.2f} МПа")
    elif choice == '2':
        ratio = predict_ratio(X_raw)
        print(f"\n📊 Результат (нейронная сеть):")
        print(f"   Рекомендуемое соотношение матрица-наполнитель: {ratio:.4f}")
    elif choice == '3':
        mod, st = predict_modulus_strength(X_raw)
        ratio = predict_ratio(X_raw)
        print(f"\n📊 Комбинированный прогноз:")
        print(f"   Модуль упругости при растяжении: {mod:.2f} ГПа")
        print(f"   Прочность при растяжении: {st:.2f} МПа")
        print(f"   Соотношение матрица-наполнитель: {ratio:.4f}")
    else:
        print("❌ Неверный выбор.")

if __name__ == '__main__':
    main()
