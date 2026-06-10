import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def plot_training_history(history, epochs, save_path):
    """Membuat dan menyimpan plot kurva loss serta akurasi."""
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(14, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss', marker='o')
    plt.plot(epochs_range, history['val_loss'], label='Val Loss', marker='s')
    plt.title('Kurva Performa Loss', fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Akurasi
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_acc'], label='Train Acc', marker='o')
    plt.plot(epochs_range, history['val_acc'], label='Val Acc', marker='s')
    plt.title('Kurva Performa Akurasi', fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.show()

def evaluate_and_plot_cm(true_labels, pred_labels, class_names, save_path, title='Confusion Matrix'):
    """Mencetak Classification Report dan menampilkan Confusion Matrix Heatmap."""
    print("\n" + "="*20 + " CLASSIFICATION REPORT " + "="*20)
    print(classification_report(true_labels, pred_labels, target_names=class_names))

    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontweight='bold', pad=12)
    plt.xlabel('Prediksi Model')
    plt.ylabel('Label Sebenarnya')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.show()