import os
from torchvision import transforms, datasets
from utils import project_path
from torch.utils.data import Dataset, DataLoader, Subset

NUM_CLASSES = 600
IMGS_PER_CLASS = 20

# [0.3625] [0.0405]


def get_default_transform(image_size=128):
    transform_train = transforms.Compose([
        transforms.Resize(size=image_size),
        transforms.Grayscale(1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.RandomRotation(5)], p=0.6),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.6),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3625], std=[0.0405])
    ])

    transform_test = transforms.Compose([
        transforms.Resize(size=image_size),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3625], std=[0.0405])
    ])

    return transform_train, transform_test


class TJU_PV600(Dataset):
    def __init__(self, transform=None):
        super().__init__()
        path_dir_train = os.path.join(project_path, 'datasets', 'TJU_PV600', "train")
        path_dir_test = os.path.join(project_path, 'datasets', 'TJU_PV600', "test")
        if transform is None:
            transform = transforms.Compose([transforms.ToTensor(), ])
        train_set = datasets.ImageFolder(path_dir_train, transform=transform)
        test_set = datasets.ImageFolder(path_dir_test, transform=transform)

        dataset = {}
        merged_label = []
        merged_data = []
        for i in range(len(train_set)):
            data, label = train_set[i]
            if label not in dataset:
                dataset[label] = [data]
            else:
                dataset[label].append(data)

        for i in range(len(test_set)):
            data, label = test_set[i]
            if label not in dataset:
                dataset[label] = [data]
            else:
                dataset[label].append(data)

        for key in dataset.keys():
            imgs = dataset[key]
            for j in range(len(imgs)):
                merged_data.append(imgs[j])
                merged_label.append(key)

        self.data = merged_data
        self.label = merged_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]
        return data, label


def getTJUDataLoader(image_size=128, batch_size=30, train_imgs_per_class=10, test_imgs_per_class=5, val_imgs_per_class=5):
    assert (train_imgs_per_class + test_imgs_per_class + val_imgs_per_class) <= IMGS_PER_CLASS
    transform_train, transform_test = get_default_transform(image_size=image_size)
    train_dataset = TJU_PV600(transform=transform_train)
    test_dataset = TJU_PV600(transform=transform_test)
    train_img_indexs = []
    test_img_indexs = []
    val_img_indexs = []
    for class_idx in range(NUM_CLASSES):
        start_idx = class_idx * IMGS_PER_CLASS
        split_index = start_idx + train_imgs_per_class
        test_split_idx = split_index + test_imgs_per_class
        end_idx = test_split_idx + val_imgs_per_class
        train_img_indexs.extend(range(start_idx, split_index))
        test_img_indexs.extend(range(split_index, test_split_idx))
        val_img_indexs.extend(range(test_split_idx, end_idx))

    final_train_set = Subset(train_dataset, train_img_indexs)
    final_test_set = Subset(test_dataset, test_img_indexs)
    final_val_set = Subset(test_dataset, val_img_indexs)

    train_loader = DataLoader(final_train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(final_test_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(final_val_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, val_loader


def compute_mean_variance(channel=1):
    train_loader = getTJUDataLoader(image_size=128, batch_size=1, train_imgs_per_class=10,
                                    test_imgs_per_class=5, val_imgs_per_class=5)[0]
    mean, std = [0. for _ in range(channel)], [0. for _ in range(channel)]
    total = 0
    for x, _ in train_loader:
        total += x.size(0)
        for c in range(channel):
            mean[c] += x[:, c, :, :].mean()
            std[c] += x[:, c, :, :].std()
    mean = [mean[i] / total for i in range(channel)]
    std = [std[i] / total for i in range(channel)]
    print(mean, std)


if __name__ == '__main__':
    compute_mean_variance()
