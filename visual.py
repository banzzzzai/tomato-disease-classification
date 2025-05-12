import os
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import random
import csv

# Параметры
model_path = './resnet50_tomato_best.pth'
dataset_dir = './PlantVillage-Tomato_split/val'
output_dir = './gradcam_results'
input_size = 224
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Создание директорий
os.makedirs(output_dir, exist_ok=True)
correct_dir = os.path.join(output_dir, 'correct')
incorrect_dir = os.path.join(output_dir, 'incorrect')
os.makedirs(correct_dir, exist_ok=True)
os.makedirs(incorrect_dir, exist_ok=True)

# Трансформации
transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Классы
class_names = sorted(os.listdir(dataset_dir))

# Модель
model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(class_names))
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Grad-CAM класс
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x, class_idx=None):
        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        loss = output[:, class_idx]
        loss.backward()
        gradients = self.gradients
        activations = self.activations
        b, k, u, v = gradients.size()
        alpha = gradients.view(b, k, -1).mean(2)
        weights = alpha.view(b, k, 1, 1)
        cam = (weights * activations).sum(1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze(0).squeeze(0).cpu().numpy()
        cam = cv2.resize(cam, (input_size, input_size))
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam, class_idx

# Grad-CAM инициализация
target_layer = model.layer4[-1]
grad_cam = GradCAM(model, target_layer)

# CSV лог
csv_path = os.path.join(output_dir, 'analysis_log.csv')
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Class', 'Image', 'Prediction', 'Correct', 'OriginalPath', 'OverlayPath'])

# Обработка изображений
for class_name in class_names:
    class_dir = os.path.join(dataset_dir, class_name)
    all_images = os.listdir(class_dir)
    random.shuffle(all_images)
    correct_found = False
    incorrect_found = False

    for img_name in all_images:
        img_path = os.path.join(class_dir, img_name)
        img_original = cv2.imread(img_path)
        if img_original is None:
            continue
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_original, (input_size, input_size))

        img_pil = Image.fromarray(img_original)
        img_tensor = transform(img_pil).unsqueeze(0).to(device)

        cam, pred_idx = grad_cam(img_tensor)
        pred_class = class_names[pred_idx]

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        overlay = heatmap + np.float32(img_resized) / 255
        overlay = overlay / np.max(overlay)
        overlay_img = np.uint8(overlay * 255)

        correct = pred_class == class_name
        if correct and not correct_found:
            save_prefix = os.path.join(correct_dir, f"{class_name}_correct")
            correct_found = True
        elif not correct and not incorrect_found:
            save_prefix = os.path.join(incorrect_dir, f"{class_name}_incorrect")
            incorrect_found = True
        else:
            continue

        original_save = save_prefix + "_original.jpg"
        overlay_save = save_prefix + "_gradcam.jpg"
        cv2.imwrite(original_save, cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
        cv2.imwrite(overlay_save, cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))

        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                class_name,
                img_name,
                pred_class,
                'Yes' if correct else 'No',
                original_save,
                overlay_save
            ])

        if correct_found and incorrect_found:
            break