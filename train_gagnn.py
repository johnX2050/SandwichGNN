import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.cluster import KMeans

from GAGNN.model import Model
from GAGNN.dataset import trainDataset, valDataset, testDataset
import argparse

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Multi-city AQI forecasting')
parser.add_argument('--device', type=str, default='cuda', help='')
parser.add_argument('--mode', type=str, default='full', help='')
parser.add_argument('--encoder', type=str, default='self', help='')
parser.add_argument('--w_init', type=str, default='rand', help='')
parser.add_argument('--mark', type=str, default='', help='')
parser.add_argument('--run_times', type=int, default=1, help='')
parser.add_argument('--epoch', type=int, default=1, help='')
parser.add_argument('--batch_size', type=int, default=64, help='')
parser.add_argument('--w_rate', type=int, default=50, help='')
parser.add_argument('--city_num', type=int, default=209, help='')
parser.add_argument('--group_num', type=int, default=15, help='')
parser.add_argument('--gnn_h', type=int, default=32, help='')
parser.add_argument('--gnn_layer', type=int, default=2, help='')
parser.add_argument('--x_em', type=int, default=32, help='x embedding')
parser.add_argument('--date_em', type=int, default=4, help='date embedding')
parser.add_argument('--loc_em', type=int, default=12, help='loc embedding')
parser.add_argument('--edge_h', type=int, default=12, help='edge h')
parser.add_argument('--lr', type=float, default=0.001, help='lr')
parser.add_argument('--wd', type=float, default=0.001, help='weight decay')
parser.add_argument('--pred_step', type=int, default=6, help='step')
parser.add_argument('--lr_decay', type=float, default=0.9, help='')
args = parser.parse_args()
print(args)

train_dataset = trainDataset()
val_dataset = valDataset()
test_dataset = testDataset()
print(len(train_dataset) + len(val_dataset) + len(test_dataset))
train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size,
                               shuffle=False, num_workers=0, pin_memory=True)
val_loader = Data.DataLoader(val_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=0, pin_memory=True)

test_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)

device = args.device
# city_index = [0,2,30,32,43]
path = './data'

for times in range(args.run_times):
    start = time.time()

    w = None
    if args.w_init == 'group':
        city_loc = np.load(os.path.join(path, 'loc_filled.npy'), allow_pickle=True)
        kmeans = KMeans(n_clusters=args.group_num, random_state=0).fit(city_loc)
        group_list = kmeans.labels_.tolist()
        w = np.random.randn(args.city_num, args.group_num)
        w = w * 0.1
        for i in range(len(group_list)):
            w[i, group_list[i]] = 1.0
        w = torch.FloatTensor(w).to(device, non_blocking=True)

    city_model = Model(args.mode, args.encoder, args.w_init, w, args.x_em, args.date_em, args.loc_em, args.edge_h,
                       args.gnn_h,
                       args.gnn_layer, args.city_num, args.group_num, args.pred_step, device).to(device)
    city_num = sum(p.numel() for p in city_model.parameters() if p.requires_grad)
    print('city_model:', 'Trainable,', city_num)
    # print(city_model)
    criterion = nn.L1Loss(reduction='sum')
    all_params = city_model.parameters()
    w_params = []
    other_params = []
    for pname, p in city_model.named_parameters():
        if pname == 'w':
            w_params += [p]
    params_id = list(map(id, w_params))
    other_params = list(filter(lambda p: id(p) not in params_id, all_params))
    # print(len(w_params),len(other_params))
    optimizer = torch.optim.Adam([
        {'params': other_params},
        {'params': w_params, 'lr': args.lr * args.w_rate}
    ], lr=args.lr, weight_decay=args.wd)

    val_loss_min = np.inf
    train_loss_list = []
    val_loss_list = []
    cnt_val_loss_grow = 0
    is_early_stop = False
    stop_train_num = 0
    for epoch in range(args.epoch):
        total_train_loss = 0
        for i, data in enumerate(train_loader):
            data = [item.to(device, non_blocking=True) for item in data]
            x, u, y, edge_index, edge_w, loc = data
            outputs = city_model(x, u, edge_index, edge_w, loc)
            loss = criterion(y, outputs)
            total_train_loss += loss.item()
            city_model.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0 and i % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch, args.epoch, i, int(len(train_dataset) / args.batch_size), loss.item()))

        if epoch % 5 == 0:
            with torch.no_grad():
                val_loss = 0
                for j, data_val in enumerate(val_loader):
                    data_val = [item.to(device, non_blocking=True) for item in data_val]
                    x_val, u_val, y_val, edge_index_val, edge_w_val, loc_val = data_val
                    outputs_val = city_model(x_val, u_val, edge_index_val, edge_w_val, loc_val)
                    batch_loss = criterion(y_val, outputs_val)
                    val_loss += batch_loss.item()
                print('Epoch:', epoch, ', val_loss:', val_loss)
                if val_loss < val_loss_min:
                    torch.save(city_model.state_dict(), args.encoder + '_para_' + args.mark + '.ckpt')
                    val_loss_min = val_loss
                    print('parameters have been updated during epoch ', epoch)
                    cnt_val_loss_grow = 0
                # else:
                #     cnt_val_loss_grow += 1
                #     if cnt_val_loss_grow >= 3:
                #         # is_early_stop = True
                #         # stop_train_num = epoch
                #         optimizer = torch.optim.Adam([
                #             {'params': other_params},
                #             {'params': w_params, 'lr': args.lr * args.w_rate * args.lr_decay}
                #             ], lr=args.lr * args.lr_decay, weight_decay=args.wd)
        if is_early_stop:
            break
        train_loss_list.append(total_train_loss / (len(train_dataset) * args.city_num * args.pred_step))
        val_loss_list.append(val_loss / (len(val_dataset) * args.city_num * args.pred_step))

    train_loss_tensor = torch.tensor(train_loss_list, device='cpu')
    val_loss_tensor = torch.tensor(val_loss_list, device='cpu')
    if is_early_stop:
        epochs = range(stop_train_num)
    else:
        epochs = range(args.epoch)
    plt.plot(epochs, train_loss_tensor, 'g', label='Training loss')
    plt.plot(epochs, val_loss_tensor, 'b', label='validation loss')
    plt.title('Training and Validation loss for new GAGNN')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim(1, 1000)

    plt.yscale('log')
    plt.legend()
    plt.grid()
    # plt.show()
    plt.savefig("loss_new_" + str(times) + ".png", dpi=200)
    plt.clf()

    mae_loss = torch.zeros(args.city_num, args.pred_step).to(device)
    rmse_loss = torch.zeros(args.city_num, args.pred_step).to(device)


    def cal_loss(outputs, y):
        global mae_loss, rmse_loss
        temp_loss = torch.abs(outputs - y)
        mae_loss = torch.add(mae_loss, temp_loss.sum(dim=0))

        temp_loss = torch.pow(temp_loss, 2)
        rmse_loss = torch.add(rmse_loss, temp_loss.sum(dim=0))


    with torch.no_grad():
        city_model.load_state_dict(torch.load(args.encoder + '_para_' + args.mark + '.ckpt'))
        w_weight = city_model.state_dict()['w']
        w_weight = F.softmax(w_weight)
        _, w_weight = torch.max(w_weight, dim=-1)
        print(w_weight.cpu().tolist())

        for i, data in enumerate(test_loader):
            data = [item.to(device, non_blocking=True) for item in data]
            x, u, y, edge_index, edge_w, loc = data
            outputs = city_model(x, u, edge_index, edge_w, loc)
            cal_loss(outputs, y)

        mae_loss = mae_loss / (len(test_dataset))
        rmse_loss = rmse_loss / (len(test_dataset))
        mae_loss = mae_loss.mean(dim=0)
        rmse_loss = rmse_loss.mean(dim=0)

        end = time.time()
        print('Running time: %s Seconds' % (end - start))

        mae_loss = mae_loss.cpu()
        rmse_loss = rmse_loss.cpu()

        print('Type:', type(mae_loss[0].item()))

        print('mae for new GAGNN:', np.array(mae_loss))
        print('rmse for new GAGNN:', np.sqrt(np.array(rmse_loss)))

        mae_loss_tensor = mae_loss.unsqueeze(dim=0)
        rmse_loss = torch.sqrt(rmse_loss)
        rmse_loss_tensor = rmse_loss.unsqueeze(dim=0)
        if times == 0:
            all_loss_tensor_mae = mae_loss_tensor
            all_loss_tensor_rmse = rmse_loss_tensor
        else:
            all_loss_tensor_mae = torch.cat([all_loss_tensor_mae, mae_loss_tensor], dim=0)
            all_loss_tensor_rmse = torch.cat([all_loss_tensor_rmse, rmse_loss_tensor], dim=0)

print("======================================================================================================")
mean_mae = all_loss_tensor_mae.mean(dim=0)
mean_rmse = all_loss_tensor_rmse.mean(dim=0)
print("The mean of NEW MAE for all " + str(args.run_times) + " times: ", mean_mae)
print("The mean of NEW RMSE for all " + str(args.run_times) + " times: ", mean_rmse)
std_mae = all_loss_tensor_mae.std(dim=0)
std_rmse = all_loss_tensor_rmse.std(dim=0)
print("The std' of NEW MAE for all " + str(args.run_times) + " times: ", std_mae)
print("The std' of NEW RMSE for all " + str(args.run_times) + " times: ", std_rmse)
print("======================================================================================================")
for i, data in enumerate(Data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)):
    data = [item.to(device, non_blocking=True) for item in data]
    x, u, y, edge_index, edge_w, loc = data
    outputs = city_model(x, u, edge_index, edge_w, loc)
    if i == 305:
        print(x[:, 0])
        print(outputs[:, 0])
