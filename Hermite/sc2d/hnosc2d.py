"""
This code is written by Ruijie Bai, you may contact us through bairuijie23@nudt.edu.cn
"""
import os
from torch.utils.data import DataLoader
from timeit import default_timer
from utilities import *
from scipy.io import loadmat
from hermitepack_sc2d import SOL2d_Vandermonde
device = torch.device('cuda:0')
torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True


data_name='sc2d1100'

import argparse
def get_args():
    parser = argparse.ArgumentParser('Spectral Operator Learning', add_help=False)

    parser.add_argument('--data-dict', default='../data/', type=str, help='dataset folder')
    parser.add_argument('--data-name', default='sc2d1100.mat', type=str, help='dataset name')  
    parser.add_argument('--epochs', default=0, type=int, help='training iterations')
    parser.add_argument('--sub', default=1, type=int, help='sub-sample on the data')
    parser.add_argument('--lr', default=4e-3, type=float, help='learning rate')
    parser.add_argument('--bw', default=1, type=int, help='band width')
    parser.add_argument('--batch-size', default=20, type=int, help='batch size')
    parser.add_argument('--step-size', default=500, type=int, help='step size for the StepLR (if used)')
    parser.add_argument('--modes', default=60, type=int, help='Fourier-like modes')
    parser.add_argument('--suffix', default='', type=str, help='')
    parser.add_argument('--triL', default=0, type=int, help='')
    parser.add_argument('--scdl', default='step', type=str, help='')

    return parser.parse_args()


args = get_args()

epochs = args.epochs  
step_size = args.step_size  
batch_size = args.batch_size  
sub = args.sub  
learning_rate = args.lr  
bandwidth = args.bw  
modes = args.modes  
suffix = args.suffix
triL = args.triL
scdl = args.scdl
skip=True

gamma = 0.5  
weight_decay = 1e-8
train_size, test_size = 1000, 100
width = 50
num_workers = 2
iterations = epochs*(train_size//batch_size)

data_PATH = args.data_dict + data_name + '.mat'
file_name = 'hermite2d-' + data_name + str(sub) + '-modes' + str(modes) + '-width' + str(width) +'float1024' +'-epochs'+ str(5000) + '-bw' + str(bandwidth) + '-triL' + str(triL) + '-' + scdl + suffix
result_PATH = '../model/' + file_name + '.pkl'

if os.path.exists(result_PATH):
    print("-" * 40 + "\nWarning: pre-trained model already exists:\n" + result_PATH + "\n" + "-" * 40)

print('data:', data_PATH)
print('result_PATH:', result_PATH)
print('batch_size', batch_size, 'learning_rate', learning_rate, 'epochs', epochs, 'bandwidth', bandwidth)
print('weight_decay', weight_decay, 'width', width, 'modes', modes, 'sub', sub, 'triL', triL)


raw_data=loadmat(data_PATH)

x_data_raw, y_data_raw = raw_data['f_sub'], raw_data['u_sub']
x_data_np = x_data_raw[:, ::sub, ::sub]
y_data_np = y_data_raw[:, ::sub, ::sub]
y_data_real_np = y_data_np.real
y_data_imag_np = y_data_np.imag
if np.iscomplexobj(x_data_np):
    print("Warning: Input data (f_sub) was complex. Taking only the real part.")
    x_data_real_np = x_data_np.real
else:
    x_data_real_np = x_data_np
x_data_imag_np = np.zeros_like(x_data_real_np)
y_data_real = torch.tensor(y_data_real_np, dtype=torch.float)
y_data_imag = torch.tensor(y_data_imag_np, dtype=torch.float)

x_data_real = torch.tensor(x_data_real_np, dtype=torch.float)
x_data_imag = torch.tensor(x_data_imag_np, dtype=torch.float)
y_data = torch.stack([y_data_real, y_data_imag], dim=-1)
x_data = torch.stack([x_data_real, x_data_imag], dim=-1)
data_size, Nx, Ny, _ = x_data.shape
x_coord = torch.tensor(raw_data['x']).squeeze()  
grid_x, grid_y = torch.meshgrid(x_coord, x_coord, indexing="ij")  
grid = torch.stack((grid_x, grid_y), dim=-1)  
grid = grid.unsqueeze(0).repeat(x_data.shape[0], 1, 1, 1)  








print('y_data dtype:', y_data.dtype)
print('grid shape:', grid.shape)



print('data size =', data_size, 'training size =', train_size, 'test size =', test_size, 'Nx =', Nx, 'Ny =', Ny)






train_loader = DataLoader(
    torch.utils.data.TensorDataset(x_data[:train_size, :, :, :], y_data[:train_size, :, :]),
    batch_size=batch_size, shuffle=True, num_workers=num_workers)

test_loader = DataLoader(
    torch.utils.data.TensorDataset(x_data[-test_size:, :, :, :], y_data[-test_size:, :, :]),
    batch_size=batch_size, shuffle=False, num_workers=num_workers)


model = SOL2d_Vandermonde(2, modes, width, bandwidth,out_channels=2, skip=skip,triL=triL).to(device).float()





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
if epochs >= 5000:
    torch.save({
        'model': model.state_dict(), 'batch_size': batch_size, 'learning_rate': learning_rate, 'epochs': epochs,
        'weight_decay': weight_decay, 'width': width, 'modes': modes, 'sub': sub,
        'loss_list': loss_list, 'train_list': train_list, 'code': current_code
    }, result_PATH)



x_data_raw, y_data_raw = raw_data['f_sub'], raw_data['u_sub']
grid = torch.tensor(raw_data['x'])
Nx, Ny = 60, 60
x_coord = grid.squeeze().numpy()
grid_x, grid_y = np.meshgrid(x_coord, x_coord, indexing="ij")

x_data_np = x_data_raw[:, ::sub, ::sub]
y_data_np = y_data_raw[:, ::sub, ::sub]
y_data_real_np = y_data_np.real
y_data_imag_np = y_data_np.imag
if np.iscomplexobj(x_data_np):
    x_data_real_np = x_data_np.real
else:
    x_data_real_np = x_data_np
x_data_imag_np = np.zeros_like(x_data_real_np)


y_data_real = torch.tensor(y_data_real_np, dtype=torch.float)
y_data_imag = torch.tensor(y_data_imag_np, dtype=torch.float)

x_data_real = torch.tensor(x_data_real_np, dtype=torch.float)
x_data_imag = torch.tensor(x_data_imag_np, dtype=torch.float)


y_data = torch.stack([y_data_real, y_data_imag], dim=-1)
x_data = torch.stack([x_data_real, x_data_imag], dim=-1)
x_test, y_test = x_data[-test_size:, :, :, :].to(device), y_data[-test_size:, :, :]
with torch.no_grad():
    y_pred = model(x_test).reshape(test_size, Nx, Ny,2).cpu()
peer_loss = LpLoss(reduction=False)
test_err = peer_loss(y_pred.reshape(y_test.shape[0], -1), y_test.reshape(y_test.shape[0], -1))
print('l2 error v.s. max error', str(test_err.sum().item()/test_size)[:20], test_err.max().item())


