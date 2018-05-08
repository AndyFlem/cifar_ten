import torch

device=torch.device('cpu')

features,samples,neurons,classes=3,4,5,1

x=torch.randn(samples, features,device=device)
y=torch.randn(samples, classes,device=device)

w1=torch.randn(features,neurons,device=device)
w2=torch.randn(neurons,classes,device=device)


learning_rate=1e-3
for t in range(1500):
    h=x.mm(w1)
    h_relu=h.clamp(min=0)
    y_pred=h_relu.mm(w2)
    loss=(y_pred-y).pow(2).sum()

    grad_y_pred=2.0 * (y_pred - y)
    grad_w2=h_relu.t().mm(grad_y_pred)
    grad_h_relu=grad_y_pred.mm(w2.t())
    grad_h=grad_h_relu.clone()
    grad_h[h<0]=0
    grad_w1=x.t().mm(grad_h)

    w1-=learning_rate*grad_w1
    w2-=learning_rate*grad_w2
    
    print("Loss: {:.6f}".format(loss.item()))
    
