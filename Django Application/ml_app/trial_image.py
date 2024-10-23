# train_model.py
import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import ViTForImageClassification, ViTFeatureExtractor
from sklearn.model_selection import train_test_split
from PIL import Image

# Custom dataset class
class DeepfakeDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Load dataset
def load_dataset(dataset_dir):
    images = []
    labels = []
    for label in ['real', 'fake']:
        label_dir = os.path.join(dataset_dir, label)
        for img_file in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_file)
            images.append(img_path)
            labels.append(0 if label == 'real' else 1)  # 0 for real, 1 for fake
    return images, labels

# Main function to train the model
def train():
    dataset_dir = './dataset'  # Path to your dataset directory
    images, labels = load_dataset(dataset_dir)
    train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Data augmentation and normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_dataset = DeepfakeDataset(train_images, train_labels, transform)
    val_dataset = DeepfakeDataset(val_images, val_labels, transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Load model
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=2)
    model.train()

    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    for epoch in range(5):  # Number of epochs
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images).logits
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    # Save the model
    model.save_pretrained('./deepfake_model')

if __name__ == '__main__':
    train()
