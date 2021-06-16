import os
os.makedirs('pt', exist_ok=True)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--cost_smoothing', type=float, default=0)
args = parser.parse_args()
cost_smoothing = args.cost_smoothing


import numpy as np
POSN_DISC, VEL_DISC, ACT_DISC = 25, 11, 11
x, y = np.linspace(-5, 5, POSN_DISC), np.linspace(-15, 15, POSN_DISC)
vx, vy = np.linspace(-2, 2, VEL_DISC), np.linspace(-2, 2, VEL_DISC)
ax, ay = np.linspace(-1, 1, ACT_DISC), np.linspace(-1, 1, ACT_DISC)


# # Training Q-network

import numpy as np
import torch
import torch.nn as nn

def mlp(sizes, activation, output_activation=nn.Identity, batch_norm=False):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers.append(nn.Linear(sizes[j], sizes[j+1]))
        if batch_norm and j < len(sizes)-2:
            layers.append(nn.BatchNorm1d(sizes[j+1]))
        layers.append(act())
    return nn.Sequential(*layers)


import torch.optim as optim
tr_criterion = nn.BCEWithLogitsLoss()
val_criterion = nn.L1Loss()



qdata = np.load(f'npz/q_smooth_{cost_smoothing}.npz')
s, a, q = qdata['s'], qdata['a'], qdata['q']
q /= qdata['q'].max()
qdataset = torch.utils.data.TensorDataset(torch.from_numpy(np.concatenate([s, a], axis=1)).float(),
                                          torch.from_numpy(q[:, np.newaxis]).float())


loader = torch.utils.data.DataLoader(qdataset, batch_size=4096, shuffle=True)


# Train two networks
for i in range(2):
    qnet = mlp([qdataset.tensors[0].shape[1], 256, 256, 256, 1], torch.nn.ReLU)
    qnet.cuda()


    optimizer = optim.Adam(qnet.parameters(), lr=1e-3)


    all_inp, all_tgt = [x.cpu() for x in qdataset.tensors]
    for epoch in range(10):
        vloss, ctr = 0., 0
        for data in loader:
            inp, tgt = [x.cuda() for x in data]
            optimizer.zero_grad()
            preds = qnet(inp)
            loss = tr_criterion(preds, tgt)
            with torch.no_grad():
                vloss += val_criterion(torch.sigmoid(preds), tgt)
                ctr += 1
            loss.backward()
            optimizer.step()

        if (epoch+1) % 1 == 0:
            print("[%d] tloss: %.3f" % (epoch+1, vloss/ctr))


    qnet.cpu()
    torch.save(qnet.state_dict(), f'pt/qnet{i+1}_smooth_{cost_smoothing}.pt')


vdata = np.load(f'npz/v_smooth_{cost_smoothing}.npz')
s, v = vdata['s'], vdata['v']
v /= vdata['v'].max()
vdataset = torch.utils.data.TensorDataset(torch.from_numpy(s).float(),
                                          torch.from_numpy(v[:, np.newaxis]).float())


loader = torch.utils.data.DataLoader(vdataset, batch_size=1024, shuffle=True)


# Train two networks
for i in range(2):
    vnet = mlp([vdataset.tensors[0].shape[1], 256, 256, 256, 1], torch.nn.ReLU)
    optimizer = optim.Adam(vnet.parameters(), lr=1e-3)


    for epoch in range(100):
        vloss, ctr = 0., 0
        for data in loader:
            inp, tgt = data
            optimizer.zero_grad()
            preds = vnet(inp)
            loss = tr_criterion(preds, tgt)
            with torch.no_grad():
                vloss += val_criterion(torch.sigmoid(preds), tgt)
                ctr += 1
            loss.backward()
            optimizer.step()

        if (epoch+1) % 5 == 0:
            print("[%d] tloss: %.3f" % (epoch+1, vloss/ctr))

    torch.save(vnet.state_dict(), f'pt/vnet{i+1}_smooth_{cost_smoothing}.pt')
