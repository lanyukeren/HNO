"""
This code is written by Ruijie Bai, you may contact us through bairuijie23@nudt.edu.cn
"""
import os
import sys
from torch.utils.data import DataLoader
from timeit import default_timer
from utilities import *
from scipy.io import loadmat
import matplotlib.pyplot as plt
from hermitepack2d1012 import SOL2d_Vandermonde
from scipy.special import eval_hermite
from scipy.interpolate import RectBivariateSpline
import torch.nn.functional as F
from math import factorial
from scipy.special import factorial

device = torch.device('cuda:0')

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

data_name='Heat2d_DataNew'

import argparse
def get_args():
    parser = argparse.ArgumentParser('Spectral Operator Learning', add_help=False)

    parser.add_argument('--data-dict', default='../data/', type=str, help='dataset folder')
    parser.add_argument('--data-name', default='Heat2d_DataNew.mat', type=str, help='dataset name')  
    parser.add_argument('--epochs', default=500, type=int, help='training iterations')
    parser.add_argument('--sub', default=1, type=int, help='sub-sample on the data')
    parser.add_argument('--lr', default=4e-3, type=float, help='learning rate')
    parser.add_argument('--bw', default=1, type=int, help='band width')
    parser.add_argument('--batch-size', default=20, type=int, help='batch size')
    parser.add_argument('--step-size', default=50, type=int, help='step size for the StepLR (if used)')
    parser.add_argument('--modes', default=50, type=int, help='Fourier-like modes')
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
file_name = 'hermite2d-' + data_name + str(sub) + '-modes' + str(modes) + '-width' + str(width) +'float' +'-epochs'+ str(5000) + '-bw' + str(bandwidth) + '-triL' + str(triL) + '-' + scdl + suffix
result_PATH = '../model/' + file_name + '.pkl'

if os.path.exists(result_PATH):
    print("-" * 40 + "\nWarning: pre-trained model already exists:\n" + result_PATH + "\n" + "-" * 40)

raw_data=loadmat(data_PATH, mat_dtype=True)
x_data, y_data = raw_data['f_sub'], raw_data['u_sub']


x_coord = torch.tensor(raw_data['x']).squeeze()
grid_x, grid_y = torch.meshgrid(x_coord, x_coord, indexing="ij")  
grid = torch.stack((grid_x, grid_y), dim=-1)
grid = grid.unsqueeze(0).repeat(x_data.shape[0], 1, 1, 1)
x_data, y_data = torch.tensor(x_data[:, ::sub, ::sub], dtype=torch.float), torch.tensor(y_data[:, ::sub, ::sub], dtype=torch.float)
data_size, Nx, Ny = x_data.shape
x_data = x_data[..., None]

train_loader = DataLoader(
    torch.utils.data.TensorDataset(x_data[:train_size, :, :, :], y_data[:train_size, :, :]),
    batch_size=batch_size, shuffle=True, num_workers=num_workers)

test_loader = DataLoader(
    torch.utils.data.TensorDataset(x_data[-test_size:, :, :, :], y_data[-test_size:, :, :]),
    batch_size=batch_size, shuffle=False, num_workers=num_workers)


model = SOL2d_Vandermonde(1, modes, width, bandwidth, skip=skip,triL=triL).to(device).float()


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
        l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        l2.backward()
        mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
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











data_PATH= '../data/Heat2d_DataNew.mat'
raw_data=loadmat(data_PATH)

x_data, y_data = raw_data['f_sub'], raw_data['u_sub']
grid = torch.tensor(raw_data['x'])
Nx, Ny = 50, 50
x_coord = grid.squeeze().numpy()
grid_x, grid_y = np.meshgrid(x_coord, x_coord, indexing="ij")
x_data = torch.tensor(x_data[:, ::sub, ::sub],dtype=torch.float)[..., None]
y_data = torch.tensor(y_data[:, ::sub, ::sub],dtype=torch.float)
x_test, y_test = x_data[-test_size:, :, :, :].to(device), y_data[-test_size:, :, :]

with torch.no_grad():
    y_pred = model(x_test).reshape(test_size, Nx, Ny).cpu()

peer_loss = LpLoss(reduction=False)

test_err = peer_loss(y_pred.reshape(y_test.shape[0], -1), y_test.reshape(y_test.shape[0], -1))
print('l2 error v.s. max error', str(test_err.sum().item()/test_size)[:20], test_err.max().item())



halt

print("\n" + "="*60)


N_points = 500
all_errors = []

domain_min = x_coord.min()
domain_max = x_coord.max()

np.random.seed(42) 
arbitrary_points_x = np.random.uniform(-50, +50, N_points)
arbitrary_points_y = np.random.uniform(-50, +50, N_points)
modes_x, modes_y = model.modes[0], model.modes[1]

norm_hermite_factor_fun = lambda n: 1./(np.pi**(0.25)*np.sqrt(2.**n)*np.sqrt(factorial(n, exact=False)))
hermite_factors = np.array([norm_hermite_factor_fun(n) for n in range(max(modes_x, modes_y))])
exp_term_x = np.exp(-arbitrary_points_x**2 / 2)
exp_term_y = np.exp(-arbitrary_points_y**2 / 2)

basis_values_x = np.zeros((modes_x, N_points))
basis_values_y = np.zeros((modes_y, N_points))

for n in range(modes_x):
    H_n = eval_hermite(n, arbitrary_points_x)
    basis_values_x[n, :] = hermite_factors[n] * H_n * exp_term_x

for m in range(modes_y):
    H_m = eval_hermite(m, arbitrary_points_y)
    basis_values_y[m, :] = hermite_factors[m] * H_m * exp_term_y


with torch.no_grad():
    for i in range(test_size):

        pred_on_grid = y_pred[i]
        true_on_grid = y_test[i]
        pred_on_grid_tensor = pred_on_grid.to(device).unsqueeze(0).unsqueeze(0)
        pred_spectral_coeffs = model.T(pred_on_grid_tensor, model.X_dims).squeeze().cpu().numpy()
        pred_at_arbitrary_points = np.einsum('nm,nk,mk->k', pred_spectral_coeffs, basis_values_x, basis_values_y)

        true_on_grid_tensor = true_on_grid.to(device).unsqueeze(0).unsqueeze(0)
        true_spectral_coeffs = model.T(true_on_grid_tensor, model.X_dims).squeeze().cpu().numpy()
        true_at_arbitrary_points = np.einsum('nm,nk,mk->k', true_spectral_coeffs, basis_values_x, basis_values_y)

        diff_norm = np.linalg.norm(pred_at_arbitrary_points - true_at_arbitrary_points)
        true_norm = np.linalg.norm(true_at_arbitrary_points)
        
        if true_norm > 1e-12: 
            relative_error = diff_norm / true_norm
            all_errors.append(relative_error)


if all_errors:
    mean_relative_error = np.mean(all_errors)
    print(f"\nExperiment finished.")
    print(f"Average Relative L2 Error on {N_points} arbitrary points: {mean_relative_error:.4e}")

