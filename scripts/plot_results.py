import matplotlib.pyplot as plt
import json
import os

RESULTS_DIR = "results"

def load_history(model_name):
    history_path = os.path.join(RESULTS_DIR, f"history_{model_name}.json")
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"No existe: {history_path}")

    with open(history_path, "r") as f:
        history = json.load(f)
    return history

def plot_history(model_name):
    history = load_history(model_name)

    acc = history["accuracy"]
    val_acc = history["val_accuracy"]
    loss = history["loss"]
    val_loss = history["val_loss"]

    epochs = range(1, len(acc) + 1)

    # Accuracy
    plt.figure()
    plt.plot(epochs, acc, label="Train Acc")
    plt.plot(epochs, val_acc, label="Val Acc")
    plt.title(f"Accuracy – {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"results/plot_accuracy_{model_name}.png", dpi=300)

    # Loss
    plt.figure()
    plt.plot(epochs, loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.title(f"Loss – {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"results/plot_loss_{model_name}.png", dpi=300)

    print(f"Gráficas generadas en carpeta results/ para {model_name}")


if __name__ == "__main__":
    for model in ["cnn_simple", "mobilenetv2", "resnet50"]:
        try:
            plot_history(model)
        except:
            pass
