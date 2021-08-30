import torch
from torch.utils.data import Dataset, DataLoader, random_split

from nucleoform.transformations import lineToNumbers, labelToTensor


def partition(dataset, ratios=[0.8, 0.1, 0.1], batch_size=64):
    '''
    Partition the dataset of class Dataset into three groups, namely
    training, validation ("dev") and test data.
    '''
    assert sum(ratios) == 1

    size = len(dataset)
    n_train = int(size * ratios[0])
    n_dev = int(size * ratios[1])
    n_test = size - n_train - n_dev
    
    train, dev, test = random_split(
        dataset=dataset,
        lengths=[n_train, n_dev, n_test],
        generator=torch.Generator().manual_seed(42))

    # Dataloader
    # Defaults to drop_last=True
    params = {
        'batch_size': 64,
        'shuffle': True,
        'drop_last': True}
    train_dl = DataLoader(train, **params)
    test_dl = DataLoader(test, batch_size=64, shuffle=True, drop_last=True)
    dev_dl = DataLoader(dev, batch_size=64, shuffle=True, drop_last=True)

    return train_dl, test_dl, dev_dl


class OligoDataset(Dataset):
    '''
    - https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
    - https://pytorch.org/docs/stable/data.html#map-style-datasets
    '''
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        idx = self.data[index]

        # x = lineToTensor(idx[1])
        x = lineToNumbers(idx[1])
        y = labelToTensor(idx[0])

        return x, y
