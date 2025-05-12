import os
import cv2
import torch
import numpy as np
from torchvision import models, datasets, transforms
from torch.nn import functional as F
from PIL import Image
from tqdm import tqdm

# === Параметры ===
model_path = "resnet50_tomato_best_amp.pth"
dataset_dir = "./PlantVillage-Tomato_split/test"
output_dir = "./gradcam_resnet50"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.makedirs(output_dir, exist_ok=True)

# === Модель: ResNet-50 ===
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()
target_layer = model.layer4[-1]

# === Grad-CAM реализация ===
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
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
        activations = self.activations[0]

        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_grad[i]

        heatmap = activations.detach().cpu().numpy().mean(axis=0)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= heatmap.max() + 1e-8
        return heatmap

# === Преобразования ===
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === Датасет ===
dataset = datasets.ImageFolder(dataset_dir, transform=transform)
class_names = dataset.classes
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

gradcam = GradCAM(model, target_layer)
samples_per_class = {}
max_correct = 1
max_incorrect = 2

# === Обработка ===
for inputs, labels in tqdm(dataloader):
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    correct = preds.item() == labels.item()

    # Восстановление оригинала
    inv_norm = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    img = inv_norm(inputs.squeeze().cpu()).clamp(0, 1).permute(1, 2, 0).numpy()
    img_bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Grad-CAM карта
    heatmap = gradcam(inputs)
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 0.5, heatmap_color, 0.5, 0)

    true_cls = class_names[labels.item()]
    pred_cls = class_names[preds.item()]
    status = "correct" if correct else "incorrect"
    key = f"{true_cls}_{status}"
    samples_per_class.setdefault(key, 0)

    if (correct and samples_per_class[key] < max_correct) or (not correct and samples_per_class[key] < max_incorrect):
        base_name = f"{true_cls}_{pred_cls}_{status}_{samples_per_class[key]}"
        out_dir = os.path.join(output_dir, key)
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, base_name + "_original.jpg"), img_bgr)
        cv2.imwrite(os.path.join(out_dir, base_name + "_gradcam.jpg"), overlay)
        samples_per_class[key] += 1

    if all(v >= max_correct if "correct" in k else v >= max_incorrect for k, v in samples_per_class.items()):
        if len(samples_per_class) >= len(class_names) * 2:
            break
