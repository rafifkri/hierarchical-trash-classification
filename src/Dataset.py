import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_transforms():
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val_test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    return data_transforms

def create_dataloaders(data_dir, batch_size=32, num_workers=2):
    t_funcs = get_transforms()
    
    phases = ['train', 'val', 'test']
    image_datasets = {}
    
    for phase in phases:
        phase_dir = os.path.join(data_dir, phase)
        if os.path.exists(phase_dir):
            image_datasets[phase] = datasets.ImageFolder(phase_dir, t_funcs['train'] if phase == 'train' else t_funcs['val_test'])
        else:
            # Mengantisipasi jika folder test tidak diletakkan di sub-branch yang sama
            print(f"[INFO] Folder {phase_dir} tidak ditemukan, loader dilewati.")
            
    dataloaders = {
        phase: DataLoader(image_datasets[phase], batch_size=batch_size, shuffle=(phase == 'train'), num_workers=num_workers)
        for phase in image_datasets.keys()
    }
    
    dataset_sizes = {phase: len(image_datasets[phase]) for phase in image_datasets.keys()}
    class_names = image_datasets['train'].classes if 'train' in image_datasets else None
    
    return dataloaders, dataset_sizes, class_names