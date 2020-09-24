from torchvision import datasets, transforms

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
dataset1 = datasets.FashionMNIST(
        root = './data/FashionMNIST',
        train = True,
        download = True,
        transform = transforms.Compose([
            transforms.ToTensor()                                 
        ])
)
dataset2 = datasets.FashionMNIST(
            root = './data/FashionMNIST',
            train = False,
            download = True,
            transform = transforms.Compose([
                transforms.ToTensor()                                 
            ])
        )