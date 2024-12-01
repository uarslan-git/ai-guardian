import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CustomDatasetQuoable(Dataset):
    def __init__(self):
        self.root_dir = './'
        self.image_paths = []
        self.labels = []
        self.classes = [
            d for d in sorted(os.listdir(self.root_dir))
            if os.path.isdir(os.path.join(self.root_dir, d))
        ]

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),  # Resize images to 128x128
            transforms.ToTensor(),         # Convert image to PyTorch tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
        ])


        for label, class_name in enumerate(self.classes):
            class_path = os.path.join(self.root_dir, class_name)

            # Add image paths and labels
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)

                # Filter only valid image files
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                    self.image_paths.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]
        label = self.labels[idx]

        img = Image.open(img_path)
        img = self.transform(img)

        return img, label

