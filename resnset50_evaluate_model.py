import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_resnet50():
    model_path = 'resnet50_tomato_best_amp.pth'
    dataset_dir = './PlantVillage-Tomato_split'
    input_size = 224
    batch_size = 128
    num_classes = 10

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Используется устройство: {device}")

    # === Трансформация тестовых изображений ===
    test_transform = transforms.Compose([
        transforms.Resize(input_size + 32),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # === Загрузка тестовой выборки ===
    test_dir = os.path.join(dataset_dir, 'test')
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=os.cpu_count() // 3, pin_memory=True
    )

    class_names = test_dataset.classes

    # === Модель ResNet-50 ===
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # === Метрики ===
    print("\n🔎 Classification Report (ResNet-50):")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    print("\n🧩 Confusion Matrix (ResNet-50):")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    evaluate_resnet50()
