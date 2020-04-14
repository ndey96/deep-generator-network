import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def get_data_tools(batch_size):
    imagenet_transforms = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder(
            '/home/torenvln/git/fastdata2/ilsvrc2012/training_images',
            imagenet_transforms)
    # tlist = list(range(0, len(ds), 300))
    # ds = torch.utils.data.Subset(ds, tlist)
    train_loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True)

    ds2 = datasets.ImageFolder(
            '/home/torenvln/git/fastdata2/ilsvrc2012/validation_images',
            imagenet_transforms)
    # vlist = list(range(0, len(ds), 10))
    # ds2 = torch.utils.data.Subset(ds, vlist)
    val_loader = torch.utils.data.DataLoader(
        ds2,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True)
    
    return imagenet_transforms, train_loader, val_loader