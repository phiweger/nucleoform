import torch


def lineToNumbers(line):
    '''
    We reserve 0 for padding.
    '''
    d = {k: v for k, v in zip('abcdef*', range(1, 8))}
    return torch.tensor([d[i] for i in line])


def numbersToLine(numbers):
    '''
    We reserve 0 for padding.
    '''
    d = {v: k for k, v in zip('abcdef*', range(1, 8))}
    return ''.join([d[i.item()] for i in numbers])


def labelToTensor(label):
    if label == 'coding':
        return torch.tensor(1)
    else:
        return torch.tensor(0)