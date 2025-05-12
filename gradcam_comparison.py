# gradcam_batch_comparison.py
import os
import cv2
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

# === Параметры ===
data_dir = Path("./test_samples")  # Папка с 20 изображениями
output_base = Path("./gradcam_comparison")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 224

# === Классы ===
class_names = ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
               'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
               'Tomato___Spider_mites', 'Tomato___Target_Spot',
               'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
               'Tomato___healthy']

# === Преобразование ===
transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === Grad-CAM реализация ===
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
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()
        cam = cv2.resize(cam, (input_size, input_size))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx

# === Модели ===
def load_models():
    models_dict = {}

    # ResNet-50
    resnet = models.resnet50(weights=None)
    resnet.fc = torch.nn.Linear(resnet.fc.in_features, len(class_names))
    resnet.load_state_dict(torch.load("resnet50_tomato_best_amp.pth", map_location=device))
    models_dict['resnet50'] = (resnet.to(device).eval(), resnet.layer4[-1])

    # EfficientNet-B0
    effnet = models.efficientnet_b0(weights=None)
    effnet.classifier[1] = torch.nn.Linear(effnet.classifier[1].in_features, len(class_names))
    effnet.load_state_dict(torch.load("efficientnet_b0_tomato_best.pth", map_location=device))
    models_dict['efficientnet_b0'] = (effnet.to(device).eval(), effnet.features[-1])

    # ConvNeXt-Tiny
    convnext = models.convnext_tiny(weights=None)
    convnext.classifier[2] = torch.nn.Linear(convnext.classifier[2].in_features, len(class_names))
    convnext.load_state_dict(torch.load("convnext_tiny_tomato_best.pth", map_location=device))
    models_dict['convnext_tiny'] = (convnext.to(device).eval(), convnext.features[-1])

    return models_dict

# === Обработка одного изображения ===
def process_image(img_path, models_dict):
    img_name = img_path.stem
    img_pil = Image.open(img_path).convert("RGB")
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    img_cv = cv2.cvtColor(np.array(img_pil.resize((input_size, input_size))), cv2.COLOR_RGB2BGR)

    for model_name, (model, target_layer) in models_dict.items():
        gradcam = GradCAM(model, target_layer)
        cam, pred_idx = gradcam(img_tensor)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_cv.astype(np.float32) / 255, 0.5, heatmap.astype(np.float32) / 255, 0.5, 0)
        overlay = np.uint8(255 * overlay / overlay.max())

        out_dir = output_base / img_name / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"{img_name}_original.jpg"), img_cv)
        cv2.imwrite(str(out_dir / f"{img_name}_gradcam.jpg"), overlay)

# === Запуск ===
if __name__ == "__main__":
    models_dict = load_models()
    images = sorted(data_dir.glob("*.jpg"))
    for img_path in images:
        process_image(img_path, models_dict)
    print("✅ Grad-CAM визуализация завершена для всех моделей.")