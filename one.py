import torch

features,samples,neurons,classes=64,1000,100,10

x=torch.randn(samples,features)
y=torch.randn(samples, classes)

model=torch.nn.Sequential(
    torch.nn.Linear(features,neurons),
    torch.nn.ReLU(),
    torch.nn.Linear(neurons,classes)
)

learning_rate=1e-2
for t in range(1500):
    y_pred=model(x)
    loss=torch.nn.functional.mse_loss(y_pred,y)

    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            param-=learning_rate * param.grad
    model.zero_grad()

    print(loss)

