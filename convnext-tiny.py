import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.optim import lr_scheduler
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

# Параметры
dataset_dir = './PlantVillage-Tomato_split'
num_classes = 10
batch_size = 128
num_epochs = 30
input_size = 224
learning_rate = 0.0005

torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Аугментация и нормализация
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size + 32),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

image_datasets = {
    x: datasets.ImageFolder(os.path.join(dataset_dir, x), data_transforms[x])
    for x in ['train', 'val']
}
dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x],
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count() // 3,
        pin_memory=True
    ) for x in ['train', 'val']
}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

# Обучение
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    writer = SummaryWriter(log_dir='runs/convnext_tiny_2.0')
    scaler = GradScaler()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}\n' + '-' * 10)
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss, running_corrects = 0.0, 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    with autocast(device_type='cuda', enabled=(phase == 'train')):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
            writer.add_scalar(f'Accuracy/{phase}', epoch_acc, epoch)
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'[INFO] Обучение завершено за {time_elapsed // 60:.0f}м {time_elapsed % 60:.0f}с')
    print(f'[INFO] Лучшая точность на валидации: {best_acc:.4f}')
    writer.close()
    model.load_state_dict(best_model_wts)
    return model

if __name__ == "__main__":
    model_ft = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    for param in model_ft.parameters():
        param.requires_grad = True
    model_ft.classifier[2] = nn.Linear(model_ft.classifier[2].in_features, num_classes)
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer_ft = optim.AdamW(model_ft.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=num_epochs)

    model_ft = train_model(model_ft, criterion, optimizer_ft, scheduler, num_epochs=num_epochs)
    torch.save(model_ft.state_dict(), 'convnext_tiny_tomato_best_2.pth')
    print("✅ Модель ConvNeXt-Tiny успешно сохранена.")
