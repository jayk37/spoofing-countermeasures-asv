import sys
sys.path.append("./"), sys.path.append("../")

import os
import yaml
import time
import torch
import models
import numpy as np
import pandas as pd
import torch.optim as optim
from torchinfo import summary
import matplotlib.pyplot as plt
import torch.nn.functional as F
from ScratchModel import ScratchModel
from test import asv_cal_accuracies, cal_roc_eer
from torch.utils.data.dataloader import DataLoader
from data import PrepASV19Dataset, PrepASV15Dataset

# A-100 GPU | Command: nvidia-smi -L (Lists BUS IDs)
GPU_BUS_ID = ["MIG-GPU-714a8f8e-44b2-93f1-e64d-d5f204c379de/1/0",
              "MIG-GPU-6ff250df-07f5-cf8e-bfdb-d56c3c464126/2/0"]
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_BUS_ID[0]

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))

    dataset = 19
    data_type = 'time_frame'
    
    if not os.path.exists('./SPOOFING_RESTSSD1DMOD_ADAM_1E-3_EPOCH100_MULTIRESHEAD/'):
        os.makedirs('./SPOOFING_RESTSSD1DMOD_ADAM_1E-3_EPOCH100_MULTIRESHEAD/')

    if data_type == 'time_frame':
        if dataset == 15:
            root_path = 'F:/ASVspoof2015/'
            train_protocol_file_path = root_path + 'CM_protocol/cm_train.trn.txt'
            dev_protocol_file_path = root_path + 'CM_protocol/cm_develop.ndx.txt'
            eval_protocol_file_path = root_path + 'CM_protocol/cm_evaluation.ndx.txt'
            train_data_path = root_path + 'data/train_6/'
            dev_data_path = root_path + 'data/dev_6/'
            eval_data_path = root_path + 'data/eval_6/'
        else:
            root_path = '/home/abrol/DATASETS/ASVSPOOF/LA/'
            train_protocol_file_path = root_path + 'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
            dev_protocol_file_path = root_path + 'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'
            eval_protocol_file_path = root_path + 'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'
            train_data_path = root_path + 'data/train_6/'
            dev_data_path   = root_path + 'data/dev_6/'
            eval_data_path  = root_path + 'data/eval_6/'
    else:
        print("Program only supports 'time_frame' data type.")
        sys.exit()

    if dataset == 15:
        train_set = PrepASV15Dataset(train_protocol_file_path, train_data_path, data_type=data_type)
    else:
        train_set = PrepASV19Dataset(train_protocol_file_path, train_data_path, data_type=data_type)
    weights = train_set.get_weights().to(device)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)

    # RawNet2 Configurations
    dir_yaml = os.path.splitext('/home/abrol/CODEBASE-DEV/INVENTORY/YAML/RAWNET_CONFIG')[0] + '.yaml'
    with open(dir_yaml, 'r') as f_yaml:
            parser1 = yaml.safe_load(f_yaml)

    # Net = models.SSDNet1D()   # Res-TSSDNet 1D
    Net = models.SSDNet1D_MOD() # Res-TSSDNet 1D (Head->MultiResCNN)
    # Net = models.DilatedNet()  # Inc-TSSDNet
    # Net = models.SSDNet2D()  # 2D-Res-TSSDNet
    # Net = models.RawNet(parser1['model'], device) # RawNet2
    # Net = models.RawWaveFormCNN_MultiRes(input_dim=(32,1,96000),num_classes=2) # Multi-Resolution CNN
    # Net = ScratchModel(json_path="/home/abrol/CODEBASE-DEV/CORE/INVENTORY/JSON/SCRATCH_NET_SPOOF_4.json",n_classes=2,embs=False) # ScratchNet
    
    Net = Net.to(device)

    num_total_learnable_params = sum(i.numel() for i in Net.parameters() if i.requires_grad)
    print('Number of learnable params: {}.'.format(num_total_learnable_params))

    optimizer = optim.Adam(Net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    loss_type = 'WCE'  # {'WCE', 'mixup'}

    print('Training data: {}, Date type: {}. Training started...'.format(train_data_path, data_type))

    num_epoch = 100
    loss_per_epoch = torch.zeros(num_epoch,)
    best_d_eer = [.09, 0]
    
    if not os.path.exists('./SPOOFING_RESTSSD1DMOD_ADAM_1E-3_EPOCH100_MULTIRESHEAD/train_log/'):
        os.makedirs('./SPOOFING_RESTSSD1DMOD_ADAM_1E-3_EPOCH100_MULTIRESHEAD/train_log/')

    log_path = './SPOOFING_RESTSSD1DMOD_ADAM_1E-3_EPOCH100_MULTIRESHEAD/train_log/'
    time_name = time.ctime()
    time_name = time_name.replace(' ', '_')
    time_name = time_name.replace(':', '_')
    f = open(log_path + time_name + '.csv', 'w+')

    for epoch in range(num_epoch):
        Net.train()
        t = time.time()
        total_loss = 0
        counter = 0
        for batch in train_loader:
            counter += 1
            samples, labels, _ = batch
            samples = samples.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            if loss_type == 'mixup':
                alpha = 0.1
                lam = np.random.beta(alpha, alpha)
                lam = torch.tensor(lam, requires_grad=False)
                index = torch.randperm(len(labels))
                samples = lam*samples + (1-lam)*samples[index, :]
                preds = Net(samples)
                labels_b = labels[index]
                loss = lam * F.cross_entropy(preds, labels) + (1 - lam) * F.cross_entropy(preds, labels_b)
            else:
                preds = Net(samples)
                loss = F.cross_entropy(preds, labels, weight=weights)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        loss_per_epoch[epoch] = total_loss/counter

        dev_accuracy, d_probs = asv_cal_accuracies(dev_protocol_file_path, dev_data_path, Net, device, data_type=data_type, dataset=dataset)
        d_eer = cal_roc_eer(d_probs, show_plot=False)
        if d_eer <= best_d_eer[0]:
            best_d_eer[0] = d_eer
            best_d_eer[1] = int(epoch)

            eval_accuracy, e_probs = asv_cal_accuracies(eval_protocol_file_path, eval_data_path, Net, device, data_type=data_type, dataset=dataset)
            e_eer = cal_roc_eer(e_probs, show_plot=False)
        else:
            e_eer = .99
            eval_accuracy = 0.00

        net_str = data_type + '_' + str(epoch) + '_' + 'ASVspoof20' + str(dataset) + '_LA_Loss_' + str(round(total_loss / counter, 4)) + '_dEER_' \
                            + str(round(d_eer * 100, 2)) + '%_eEER_' + str(round(e_eer * 100, 2)) + '%.pth'
        torch.save({'epoch': epoch, 'model_state_dict': Net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss_per_epoch}, ('./SPOOFING_RESTSSD1DMOD_ADAM_1E-3_EPOCH100_MULTIRESHEAD/' + net_str))

        elapsed = time.time() - t

        print_str = 'Epoch: {}, Elapsed: {:.2f} mins, lr: {:.3f}e-3, Loss: {:.4f}, d_acc: {:.2f}%, e_acc: {:.2f}%, ' \
                    'dEER: {:.2f}%, eEER: {:.2f}%, best_dEER: {:.2f}% from epoch {}.'.\
                    format(epoch, elapsed/60, optimizer.param_groups[0]['lr']*1000, total_loss / counter, dev_accuracy * 100,
                           eval_accuracy * 100, d_eer * 100, e_eer * 100, best_d_eer[0] * 100, int(best_d_eer[1]))
        print(print_str)
        df = pd.DataFrame([print_str])
        df.to_csv(log_path + time_name + '.csv', sep=' ', mode='a', header=False, index=False)

        scheduler.step()

    f.close()
    plt.plot(torch.log10(loss_per_epoch))
    plt.show()