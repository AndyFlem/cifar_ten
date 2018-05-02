import torch

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

cifar=unpickle("C:\\Users\\user\\Documents\\ML\\cifar-10-batches-py\\data_batch_1")

N,D_in,H,D_out=64,1000,100,10
x=torch.randn(N, D_in)
y=torch.randn(N, D_out)

model=torch.nn.Sequential(
    torch.nn.Linear(D_in,H),
    torch.nn.ReLU(),
    torch.nn.Linear(H,D_out)
)

learning_rate=1e-2
for t in range(500):
    y_pred=model(x)
    loss=torch.nn.functional.mse_loss(y_pred,y)

    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            param-=learning_rate * param.grad
    model.zero_grad()

    print(loss)

print(cifar.keys())
print(cifar[b'batch_label'].decode('utf-8'))

