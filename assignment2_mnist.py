"""
Assignment 2 - MNIST Learning Strategy Experiments
Based on the lecturer's MNIST example (Learning Strategy slides, pp. 19-23).

Experiments:
2(a) L1 regularization
2(b) L2 regularization
2(c) Dropout
2(d) Early stopping
2(e) L1 + Dropout

Run:
    pip install tensorflow matplotlib pandas
    python assignment2_mnist.py

The script saves:
    results/<scenario>_summary.txt
    results/<scenario>_history.csv
    results/<scenario>_loss.png
    results/<scenario>_accuracy.png
    results/<scenario>_model_summary.txt
    results/all_results.csv
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Lecturer's architecture:
# Flatten -> Dense(64, relu) -> Dense(10, softmax)
EPOCHS = 10
BATCH_SIZE = 100

# User-defined regularization/dropout rates.
L1_RATE = 0.001
L2_RATE = 0.001
DROPOUT_RATE = 0.20
EARLY_STOP_PATIENCE = 3


def load_mnist():
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    # Same preprocessing as the lecturer's slides:
    # reshape to 28x28x1, float32, scale to [0,1], one-hot labels.
    X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    return X_train, y_train, X_test, y_test


def build_model(scenario):
    model = keras.Sequential(name=scenario)

    model.add(layers.Flatten(input_shape=(28, 28, 1)))

    if scenario == "L1":
        model.add(layers.Dense(
            64, activation="relu",
            kernel_regularizer=regularizers.l1(L1_RATE)
        ))
    elif scenario == "L2":
        model.add(layers.Dense(
            64, activation="relu",
            kernel_regularizer=regularizers.l2(L2_RATE)
        ))
    elif scenario == "Dropout":
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dropout(DROPOUT_RATE))
    elif scenario == "EarlyStopping":
        model.add(layers.Dense(64, activation="relu"))
    elif scenario == "L1_Dropout":
        model.add(layers.Dense(
            64, activation="relu",
            kernel_regularizer=regularizers.l1(L1_RATE)
        ))
        model.add(layers.Dropout(DROPOUT_RATE))
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    model.add(layers.Dense(10, activation="softmax"))
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def save_model_summary(model, scenario):
    path = os.path.join(RESULTS_DIR, f"{scenario}_model_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))


def plot_history(history, scenario):
    hist = history.history
    epochs = range(1, len(hist["loss"]) + 1)

    # Loss chart
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, hist["loss"], label="Training loss")
    plt.plot(epochs, hist["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{scenario}: Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{scenario}_loss.png"), dpi=180)
    plt.close()

    # Accuracy chart
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, hist["accuracy"], label="Training accuracy")
    plt.plot(epochs, hist["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{scenario}: Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{scenario}_accuracy.png"), dpi=180)
    plt.close()

    pd.DataFrame(hist).to_csv(
        os.path.join(RESULTS_DIR, f"{scenario}_history.csv"), index=False
    )


def run_experiment(scenario, X_train, y_train, X_test, y_test):
    tf.keras.backend.clear_session()
    model = build_model(scenario)
    save_model_summary(model, scenario)

    cbs = []
    if scenario == "EarlyStopping":
        cbs.append(callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ))

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=cbs,
        verbose=2
    )

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    train_loss = history.history["loss"][-1]
    train_accuracy = history.history["accuracy"][-1]
    val_loss = history.history["val_loss"][-1]
    val_accuracy = history.history["val_accuracy"][-1]

    best_epoch = int(np.argmin(history.history["val_loss"])) + 1

    summary = {
        "scenario": scenario,
        "epochs_ran": len(history.history["loss"]),
        "best_epoch_by_val_loss": best_epoch,
        "train_loss_last": train_loss,
        "train_accuracy_last": train_accuracy,
        "val_loss_last": val_loss,
        "val_accuracy_last": val_accuracy,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
    }

    with open(os.path.join(RESULTS_DIR, f"{scenario}_summary.txt"), "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    plot_history(history, scenario)
    return summary


def main():
    X_train, y_train, X_test, y_test = load_mnist()

    scenarios = ["L1", "L2", "Dropout", "EarlyStopping", "L1_Dropout"]
    all_results = []

    print("MNIST shapes:")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print()

    for scenario in scenarios:
        print("=" * 70)
        print("Running:", scenario)
        result = run_experiment(
            scenario, X_train, y_train, X_test, y_test
        )
        all_results.append(result)
        print(result)

    pd.DataFrame(all_results).to_csv(
        os.path.join(RESULTS_DIR, "all_results.csv"), index=False
    )

    print("\nFinished. Open results/all_results.csv and the PNG charts.")


if __name__ == "__main__":
    main()
