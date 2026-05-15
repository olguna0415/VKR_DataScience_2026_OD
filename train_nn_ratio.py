import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import joblib
from prepare_data import load_and_prepare_data

def build_nn(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    return model

def train_nn():
    (X_train, X_test, _, _, _, _, y_ratio_train, y_ratio_test, scaler) = load_and_prepare_data()
    input_dim = X_train.shape[1]
    model = build_nn(input_dim)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse', metrics=['mae', 'mse'])

    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=1e-6)
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

    history = model.fit(X_train, y_ratio_train,
                        validation_data=(X_test, y_ratio_test),
                        epochs=300, batch_size=32,
                        callbacks=[reduce_lr, early_stop],
                        verbose=1)

    model.save('ratio_nn.h5')
    joblib.dump(scaler, 'scaler.pkl')
    print("✅ Нейронная сеть обучена и сохранена (ratio_nn.h5).")

if __name__ == '__main__':
    train_nn()
