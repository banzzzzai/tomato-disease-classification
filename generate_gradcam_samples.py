import os
import cv2
import torch
import numpy as np
from torchvision import models, transforms, datasets
from torch.nn import functional as F
from PIL import Image
from tqdm import tqdm

# === Параметры ===
model_path = "efficientnet_b0_tomato_best.pth"  # путь к твоей модели
dataset_dir = "./PlantVillage-Tomato_split/test"
output_dir = "./gradcam_analysis"
model_arch = "efficientnet_b0"  # или 'resnet50', 'convnext_tiny'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.makedirs(output_dir, exist_ok=True)

# === Загрузка модели ===
if model_arch == "efficientnet_b0":
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 10)
    target_layer = model.features[-1]
elif model_arch == "resnet50":
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    target_layer = model.layer4[-1]
elif model_arch == "convnext_tiny":
    model = models.convnext_tiny(weights=None)
    model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, 10)
    target_layer = model.features[-1]
else:
    raise ValueError("Unsupported architecture")

model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# === Grad-CAM реализация ===
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hook()

    def hook(self):
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, _, __, output):
        self.activations = output

    def save_gradient(self, _, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        loss = output[:, class_idx]
        loss.backward()

        pooled_grad = torch.mean(self.gradients, dim=[0, 2, 3])
        activation = self.activations[0]

        for i in range(activation.shape[0]):
            activation[i, :, :] *= pooled_grad[i]

        heatmap = activation.detach().cpu().numpy().mean(axis=0)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= (heatmap.max() + 1e-8)
        return heatmap

# === Трансформации ===
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === Данные ===
dataset = datasets.ImageFolder(dataset_dir, transform=transform)
class_names = dataset.classes
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

gradcam = GradCAM(model, target_layer)

# === Сохранение примеров ===
samples_per_class = {}
max_correct = 1
max_incorrect = 2

for inputs, labels in tqdm(dataloader):
    inputs = inputs.to(device)
    labels = labels.to(device)
    output = model(inputs)
    _, pred = torch.max(output, 1)
    correct = (pred == labels).item()

    img_tensor = inputs.squeeze().detach().cpu()
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    img = inv_normalize(img_tensor).clamp(0, 1).permute(1, 2, 0).numpy()
    img_bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    heatmap = gradcam(inputs)
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 0.5, heatmap_color, 0.5, 0)

    label_name = class_names[labels.item()]
    pred_name = class_names[pred.item()]
    correctness = "correct" if correct else "incorrect"
    key = f"{label_name}_{correctness}"

    samples_per_class.setdefault(key, 0)

    if (correct and samples_per_class[key] < max_correct) or (not correct and samples_per_class[key] < max_incorrect):
        base_name = f"{label_name}_{pred_name}_{correctness}_{samples_per_class[key]}"
        out_dir = os.path.join(output_dir, key)
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, base_name + "_original.jpg"), img_bgr)
        cv2.imwrite(os.path.join(out_dir, base_name + "_gradcam.jpg"), overlay)
        samples_per_class[key] += 1

    if all(v >= max_correct if "correct" in k else v >= max_incorrect for k, v in samples_per_class.items()):
        if len(samples_per_class) >= len(class_names) * 2:
            break
