"""
This code is written by Ruijie Bai, you may contact us through bairuijie23@nudt.edu.cn
"""
import os
from torch.utils.data import DataLoader
from timeit import default_timer
from utilities import *
from scipy.io import loadmat
from hermite_pack import SOL1d_Vandermonde
device = torch.device('cuda:0')

torch.manual_seed(0)
np.random.seed(0)

torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

data_name = 'sc128'

import argparse
def get_args():
    parser = argparse.ArgumentParser('Spectral Operator Learning', add_help=False)

    parser.add_argument('--data-dict', default='../data/', type=str, help='dataset folder')
    parser.add_argument('--data-name', default='sc128.m', type=str, help='dataset name')
    parser.add_argument('--epochs', default=0, type=int, help='training iterations')
    parser.add_argument('--sub', default=1, type=int, help='sub-sample on the data')
    parser.add_argument('--lr', default=4e-3, type=float, help='learning rate')
    parser.add_argument('--bw', default=1, type=int, help='band width')
    parser.add_argument('--batch-size', default=20, type=int, help='batch size')
    parser.add_argument('--step-size', default=50, type=int, help='step size for the StepLR (if used)')
    parser.add_argument('--modes', default=128, type=int, help='Fourier-like modes')
    parser.add_argument('--suffix', default='', type=str, help='')
    parser.add_argument('--triL', default=0, type=int, help='')
    parser.add_argument('--scdl', default='step', type=str, help='')

    return parser.parse_args()


args = get_args()

epochs =  args.epochs  
step_size = args.step_size  
batch_size = args.batch_size  
sub = args.sub  
learning_rate = args.lr  
bandwidth = args.bw  
modes = args.modes
suffix = args.suffix
triL = args.triL
scdl = args.scdl

gamma = 0.5  
weight_decay = 1e-8
train_size, test_size = 1000, 100
width = 50
num_workers = 0



data_PATH = args.data_dict + data_name + '.mat'
file_name = 'hermite-' + data_name + str(sub)  + '-modes' + str(modes)  + '-width' + str(width) + '-bw' + str(bandwidth) + '-triL' + str(triL)  + '-' + scdl + suffix
result_PATH = '../model/' + file_name + '.pkl'

if os.path.exists(result_PATH):
    print("-"*40+"\nWarning: pre-trained model already exists:\n"+result_PATH+"\n"+"-"*40)

print('data:', data_PATH)
print('result_PATH:', result_PATH)
print('batch_size', batch_size, 'learning_rate', learning_rate, 'epochs', epochs, 'bandwidth', bandwidth)
print('weight_decay', weight_decay, 'width', width, 'modes', modes, 'sub', sub, 'triL', triL)

raw_data=loadmat(data_PATH)
x_data, y_data = raw_data['f_sub'], raw_data['u_sub']








print(y_data.dtype)
grid = torch.tensor(raw_data['x'])

x_data, y_data = torch.tensor(x_data[:, ::sub]), torch.tensor(y_data[:, ::sub])


y_data_real = y_data.real.clone().detach()  

y_data_imag = y_data.imag.clone().detach()  

y_data = torch.stack([y_data_real, y_data_imag], dim=-1)  

print(y_data.dtype)
print(x_data.shape)
print(y_data.shape)
data_size, Nx = x_data.shape


print('x_data shape:', x_data.shape)
print('grid shape:', grid.shape)
x_data_real = x_data.clone().detach()  
x_data_imag = torch.zeros_like(x_data)  


x_data = torch.stack([x_data_real, x_data_imag], dim=-1)  

print('data size = ', data_size, 'training size = ', train_size, 'test size = ', test_size, 'Nx = ', Nx)




print('x_data shape:', x_data.shape)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_data[:train_size, :, :], y_data[:train_size, :]), num_workers = num_workers,
        batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_data[-test_size:, :, :], y_data[-test_size:, :]), num_workers = num_workers,
        batch_size=batch_size, shuffle=False)


model = SOL1d_Vandermonde(2, modes, width, bandwidth, out_channels=2, triL=triL).to(device).double()

if epochs == 0:  
    print('pretrained model:' + result_PATH + ' loaded!')
    loader = torch.load(result_PATH)
    model.load_state_dict(loader['model'])
    loss_list = loader['loss_list']
    print(loader['loss_list'][-1])
print('model parameters number =', count_params(model))



from Adam import Adam

optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
if scdl == 'step':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
else:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, threshold=1e-1, patience=30, verbose=True)

train_list, loss_list = [], []
t1 = default_timer()

myloss = LpLoss(size_average=False)
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse, train_l2 = 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)

        mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
        
        l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        l2.backward()

        optimizer.step()
        train_mse += mse.item()
        train_l2 += l2.item()

    train_mse /= len(train_loader)
    train_l2 /= train_size
    train_list.append(train_l2)

    if scdl == 'step':
        scheduler.step()
    else:
        scheduler.step(train_l2)

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

    test_l2 /= test_size

    loss_list.append(test_l2)

    t2 = default_timer()
    if (ep + 1) % 100 == 0 or (ep < 30):
        print(ep, str(t2 - t1)[:4], optimizer.state_dict()['param_groups'][0]['lr'],
              train_mse, train_l2, test_l2)


import inspect

current_code = inspect.getsource(inspect.currentframe())



x_sub, y_sub = x_data, y_data


data_PATH= '../data/sc128.mat'
raw_data=loadmat(data_PATH)
x_data, y_data = raw_data['f_sub'], raw_data['u_sub']
x_data, y_data = torch.tensor(x_data[:, ::sub]), torch.tensor(y_data[:, ::sub])
y_data_real = y_data.real.clone().detach()  
y_data_imag = y_data.imag.clone().detach()  
y_data = torch.stack([y_data_real, y_data_imag], dim=-1)  
print(y_data.dtype)
print(x_data.shape)
print(y_data.shape)
x_data_real = x_data.clone().detach()  
x_data_imag = torch.zeros_like(x_data)  

x_data = torch.stack([x_data_real, x_data_imag], dim=-1)  


grid = torch.tensor(raw_data['x'])
Nx = 128





xx, y = x_data[-test_size:, ...].to(device), y_data[-test_size:, :] 



with torch.no_grad():
    yy = model(xx).cpu()
peer_loss = LpLoss(reduction=False)
test_err = peer_loss(yy.view(y.shape[0], -1), y.view(y.shape[0], -1))
print('l2 error v.s. max error', str(test_err.sum().item()/test_size)[:20], test_err.max().item())
