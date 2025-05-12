import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2
import os
from PIL import Image
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

# === Параметры ===
model_path = './convnext_tiny_tomato_best.pth'  # путь к сохранённой модели
dataset_dir = './PlantVillage-Tomato_split/test'  # путь к тестовому датасету
output_dir = './gradcam_results_convnext'  # куда сохранять изображения
input_size = 224
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Трансформации ===
transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === Классы ===
class_names = sorted(os.listdir(dataset_dir))

# === Модель ===
model = convnext_tiny(weights=None)
model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, len(class_names))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# === Grad-CAM класс ===
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x, class_idx=None):
        output = self.model(x)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        gradients = self.gradients
        activations = self.activations

        b, k, u, v = gradients.size()
        alpha = gradients.view(b, k, -1).mean(2)
        weights = alpha.view(b, k, 1, 1)

        cam = (weights * activations).sum(1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (input_size, input_size))
        cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam, class_idx

# === Инициализация Grad-CAM ===
target_layer = model.features[-1]  # Последний блок ConvNeXt
grad_cam = GradCAM(model, target_layer)

# === Обработка одного изображения ===
def process_image(img_path, save_prefix):
    # Исходное изображение
    img_cv = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (input_size, input_size))

    # Для модели
    img_pil = Image.fromarray(img_rgb)
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    # Grad-CAM
    cam, pred_idx = grad_cam(img_tensor)
    pred_class = class_names[pred_idx]

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = np.float32(heatmap) / 255 + np.float32(img_resized) / 255
    overlay = np.uint8(255 * overlay / np.max(overlay))

    # Сохранение
    os.makedirs(output_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(img_path))[0]
    cv2.imwrite(f'{output_dir}/{save_prefix}_{name}_gradcam.jpg', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f'{output_dir}/{save_prefix}_{name}_original.jpg', cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
    print(f"✅ {name} — {class_names[pred_idx]}")

# === Пример использования ===
if __name__ == "__main__":
    # Выбрать 1 изображение из каждого класса (можно заменить на свои)
    for cls in class_names:
        cls_path = os.path.join(dataset_dir, cls)
        img_file = sorted(os.listdir(cls_path))[0]
        img_path = os.path.join(cls_path, img_file)
        process_image(img_path, save_prefix=cls)
