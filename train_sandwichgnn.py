import time
import torch.nn as nn
import torch.utils.data as Data
import random

from SandwichGNN.sandwich_model import SandwichGNN
from SandwichGNN.dataset_meta_la import trainDataset_metr_la, valDataset_metr_la, testDataset_metr_la
from util import *
from einops import repeat
import argparse


def seed_torch(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"Random seed set as {seed}")

seed_torch(2023)

parser = argparse.ArgumentParser(description='ST forecasting')
parser.add_argument('--device', type=str, default='cuda', help='')
parser.add_argument('--mode', type=str, default='full', help='')
parser.add_argument('--encoder', type=str, default='self', help='')
parser.add_argument('--w_init', type=str, default='rand', help='')
parser.add_argument('--run_times', type=int, default=1, help='')
parser.add_argument('--epoch', type=int, default=1, help='')
parser.add_argument('--batch_size', type=int, default=64, help='')
parser.add_argument('--w_rate', type=int, default=50, help='')
parser.add_argument('--n_nodes', type=int, default=207, help='')
parser.add_argument('--window_size', type=list, default=[2], help='')
parser.add_argument('--s_factor', type=int, default=4, help='')
parser.add_argument('--n_heads', type=int, default=4, help='')
parser.add_argument('--n_blocks', type=int, default=3, help='')
parser.add_argument('--d_x', type=int, default=8, help='')
parser.add_argument('--d_edge', type=int, default=4, help='')
parser.add_argument('--d_model', type=int, default=8, help='')
parser.add_argument('--d_ff', type=int, default=8, help='')
parser.add_argument('--x_em', type=int, default=8, help='x embedding')
parser.add_argument('--seq_len', type=int, default=96, help='step')
parser.add_argument('--pred_len', type=int, default=12, help='step')
parser.add_argument('--lr', type=float, default=0.001, help='lr')
parser.add_argument('--wd', type=float, default=0.001, help='weight decay')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
parser.add_argument('--mark', type=str, default='', help='')
parser.add_argument('--adj_data', type=str,
                    default='/home/hjl/deep_learning_workspace/SandwichGNN/data/adj_mx.pkl', help='adj file')
args = parser.parse_args()
print(args)


predefined_A = load_adj(args.adj_data)
predefined_A = torch.tensor(predefined_A)-torch.eye(args.n_nodes)
predefined_A = predefined_A.to(args.device)

train_dataset = trainDataset_metr_la()
val_dataset = valDataset_metr_la()
test_dataset = testDataset_metr_la()

print(len(train_dataset) + len(val_dataset) + len(test_dataset))
train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size,
                               shuffle=True, num_workers=0, pin_memory=True)
val_loader = Data.DataLoader(val_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=0, pin_memory=True)

test_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)


path = 'data'

for times in range(args.run_times):
    start = time.time()

    Model = SandwichGNN(d_x=args.d_x, d_edge=args.d_edge, d_model=args.d_model, d_ff=args.d_ff,
                        seq_len=args.seq_len, pred_len=args.pred_len, n_nodes=args.n_nodes, n_blocks=args.n_blocks,
                        window_size=args.window_size, s_factor=args.s_factor, dropout=args.dropout
                        ).to(args.device)
    city_num = sum(p.numel() for p in Model.parameters() if p.requires_grad)
    print('city_model:', 'Trainable,', city_num)
    # print(city_model)
    criterion = nn.L1Loss(reduction='sum')
    all_params = Model.parameters()

    # Adam optimizer
    optimizer = torch.optim.Adam([
        {'params': all_params}
    ], lr=args.lr, weight_decay=args.wd)

    val_loss_min = np.inf
    for epoch in range(args.epoch):
        total_train_loss = 0
        batch_predefined_A = repeat(predefined_A, 'n1 n2 -> c n1 n2', c=args.batch_size)

        for i, data in enumerate(train_loader):
            data = [item.to(args.device, non_blocking=True) for item in data]
            x, y, loc = data

            # for the final batch, the batch size is not normal as usual
            if i == len(train_loader)-1:
                cur_batch_size = x.shape[0]
                batch_predefined_A = repeat(predefined_A, 'n1 n2 -> b n1 n2', b=cur_batch_size)

            outputs = Model(x, batch_predefined_A, loc)
            loss = criterion(y, outputs)
            total_train_loss += loss.item()
            Model.zero_grad()
            loss.backward()
            optimizer.step()


            if epoch % 5 == 0 and i % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch, args.epoch, i, int(len(train_dataset) / args.batch_size), loss.item()))


        if epoch % 5 == 0:
            with torch.no_grad():
                val_loss = 0

                batch_predefined_A = repeat(predefined_A, 'n1 n2 -> b n1 n2', b=args.batch_size)

                for j, data_val in enumerate(val_loader):
                    data_val = [item.to(args.device, non_blocking=True) for item in data_val]
                    x_val, y_val, loc_val = data_val

                    # for the final batch, the batch size is not normal as usual
                    if j == (len(val_loader)-1):
                        cur_batch_size = x_val.shape[0]
                        batch_predefined_A = repeat(predefined_A, 'n1 n2 -> b n1 n2', b=cur_batch_size)

                    outputs_val = Model(x_val, batch_predefined_A, loc_val)
                    batch_loss = criterion(y_val, outputs_val)
                    val_loss += batch_loss.item()

                print('Epoch:', epoch, ', val_loss:', val_loss)
                if val_loss < val_loss_min:
                    torch.save(Model.state_dict(), args.encoder + '_para_' + args.mark + '.ckpt')
                    val_loss_min = val_loss
                    print('parameters have been updated during epoch ', epoch)

        print('Train epoch %d: %f seconds'%(epoch, (time.time() - start)))
        start = time.time()

    mae_loss = torch.zeros(args.n_nodes, args.pred_len).to(args.device)
    rmse_loss = torch.zeros(args.n_nodes, args.pred_len).to(args.device)


    def cal_loss(outputs, y):
        global mae_loss, rmse_loss
        temp_loss = torch.abs(outputs - y)
        mae_loss = torch.add(mae_loss, temp_loss.sum(dim=0))

        temp_loss = torch.pow(temp_loss, 2)
        rmse_loss = torch.add(rmse_loss, temp_loss.sum(dim=0))


    with torch.no_grad():
        Model.load_state_dict(torch.load(args.encoder + '_para_' + args.mark + '.ckpt'))

        batch_predefined_A = repeat(predefined_A, 'n1 n2 -> b n1 n2', b=args.batch_size)

        for i, data in enumerate(test_loader):
            data = [item.to(args.device, non_blocking=True) for item in data]
            x, y, loc = data

            # for the final batch, the batch size is not normal as usual
            if i == len(test_loader)-1:
                cur_batch_size = x.shape[0]
                batch_predefined_A = repeat(predefined_A, 'n1 n2 -> b n1 n2', b=cur_batch_size)

            outputs = Model(x, batch_predefined_A, loc)
            cal_loss(outputs, y)

        mae_loss = mae_loss / (len(test_dataset))
        rmse_loss = rmse_loss / (len(test_dataset))
        mae_loss = mae_loss.mean(dim=0)
        rmse_loss = rmse_loss.mean(dim=0)


        end = time.time()
        print('Running time: %s Seconds' % (end - start))

        mae_loss = mae_loss.cpu()
        rmse_loss = rmse_loss.cpu()
        final_result = (mae_loss.sum() / 6).item()

        print('mae for new SandwichGNN:', np.array(mae_loss))
        print('rmse for new SanwichGNN:', np.sqrt(np.array(rmse_loss)))

        mae_loss_tensor = mae_loss.unsqueeze(dim=0)
        rmse_loss = torch.sqrt(rmse_loss)
        rmse_loss_tensor = rmse_loss.unsqueeze(dim=0)
        if times == 0:
            all_loss_tensor_mae = mae_loss_tensor
            all_loss_tensor_rmse = rmse_loss_tensor
        else:
            all_loss_tensor_mae = torch.cat([all_loss_tensor_mae, mae_loss_tensor], dim=0)
            all_loss_tensor_rmse = torch.cat([all_loss_tensor_rmse, rmse_loss_tensor], dim=0)

# print MAE and RMSE for all run times
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

        # for i, data in enumerate(Data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)):
        #     data = [item.to(device, non_blocking=True) for item in data]
        #     x, u, y, edge_index, edge_w, loc = data
        #     outputs = city_model(x, u, edge_index, edge_w, loc)
        #     if i == 305:
        #         print(x[:, 0])
        #         print(outputs[:, 0])
