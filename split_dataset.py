import os
import random
import shutil

# === Параметры ===
dataset_name = 'PlantVillage-Tomato'
split_ratios = {'train': 0.7, 'val': 0.2, 'test': 0.1}
random_seed = 42

random.seed(random_seed)

# === Директории ===
project_root = os.getcwd()
source_dir = os.path.join(project_root, dataset_name)
split_dir = os.path.join(project_root, dataset_name + '_split')

train_dir = os.path.join(split_dir, 'train')
val_dir = os.path.join(split_dir, 'val')
test_dir = os.path.join(split_dir, 'test')

for dir_path in [train_dir, val_dir, test_dir]:
    os.makedirs(dir_path, exist_ok=True)

# === Список классов ===
class_folders = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

print(f"[INFO] Найдено классов: {len(class_folders)}")
print(f"[INFO] Классы: {class_folders}")

def copy_files(file_list, src_class_dir, dest_class_dir):
    os.makedirs(dest_class_dir, exist_ok=True)
    for filename in file_list:
        src_file = os.path.join(src_class_dir, filename)
        dest_file = os.path.join(dest_class_dir, filename)
        shutil.copy2(src_file, dest_file)

# === Разделение по классам ===
for class_name in class_folders:
    class_path = os.path.join(source_dir, class_name)
    images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
    random.shuffle(images)

    total = len(images)
    train_end = int(total * split_ratios['train'])
    val_end = train_end + int(total * split_ratios['val'])

    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]

    copy_files(train_imgs, class_path, os.path.join(train_dir, class_name))
    copy_files(val_imgs, class_path, os.path.join(val_dir, class_name))
    copy_files(test_imgs, class_path, os.path.join(test_dir, class_name))

print("✅ Датасет успешно разделён!")
print(f"Train: {train_dir}")
print(f"Val: {val_dir}")
print(f"Test: {test_dir}")
