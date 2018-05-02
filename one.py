import torch


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

cifar=unpickle("C:\\Users\\Andy\\Documents\\ML\\cifar-10-batches-py\\data_batch_1")

print(cifar.keys())
print(cifar[b'batch_label'].decode('utf-8'))

