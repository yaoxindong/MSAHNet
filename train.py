from glob import glob
import parameter
import torch
import random
import manager
import numpy as np
import os
import datetime
from utils import storage
from timm.utils import NativeScaler
from evaluate import test_single_volume
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss


def main():

    ######### parser  ###########
    args = parameter.Parameter().args.parse_args()

    ######### path settings  ###########
    code_path = os.path.dirname(os.path.abspath(__file__))
    pro_path = os.path.join(code_path, '../')
    save_path = os.path.join(pro_path, 'save')

    save = storage(save_path, args.net)
    save.log("************************ project start ******************************\n")
    save.log("args: \n")
    save.log(str(args) + '\n')

    ######### Set GPUs  ###########
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu 
    torch.backends.cudnn.benchmark = True       
    torch.backends.cudnn.deterministic = False

    ########## Set Seeds ###########
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    ######### net model ###########
    net, netargsstrs = manager.get_net(args)

    save.log(str(net))
    save.log(netargsstrs)

    net_name = args.net
    logstr = '------You choose :'+ net_name +' net !------'
    save.log(logstr)

    ######### Optimizer ###########
    start_epoch = 1
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr_initial, momentum=0.9, weight_decay=0.0001)
    #optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr_initial, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.02)

    ######### Loss  ###########
    cross_entropy_loss = CrossEntropyLoss()
    dice_loss = manager.DiceLoss(args.num_classes)

    ######### DataParallel ###########
    if len(args.gpu) >= 2:
        net = torch.nn.DataParallel (net)
    net.cuda()


    ######### DataLoader ###########
    transforms = manager.get_transforms(patch_size=args.patch_size)
    train_dataset = manager.get_training_data(args.dataset, args.data_path, transforms)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, 
                            num_workers=args.train_workers, pin_memory=True, drop_last=False)

    val_dataset = manager.get_validation_data(args.dataset, args.data_path)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, 
                            num_workers=1, pin_memory=False, drop_last=False)    

    len_trainset = len(train_dataset)
    len_valset = len(val_dataset)
    print("Sizeof training set: ", len_trainset,", sizeof validation set: ", len_valset)

    ######## train ########
    
    loss_scaler = NativeScaler()
    torch.cuda.empty_cache()

    min_loss = 99999
    max_iterations = args.epoch * len(train_loader)
    max_mean_dice = 0
    for epoch in range(start_epoch, args.epoch + 1):
        times = datetime.datetime.now()
        logstr = 'Training epoch %d / %d,time: %04d-%02d-%02d_%02d-%02d-%02d' % (epoch, args.epoch, times.year, times.month, times.day, times.hour, times.minute, times.second)
        save.log(logstr)

        # training
        net.train()
        loss = 0
        loss_dice = 0
        loss_ce = 0
        lr_ = 0

        for i, data in enumerate(train_loader, 0): 

            optimizer.zero_grad()

            image = data['image'].cuda()
            label = data['label'].cuda()
            with torch.cuda.amp.autocast():
            
                output = net(image)
                loss_ce = cross_entropy_loss(output, label[:].long())
                loss_dice = dice_loss(output, label, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = args.lr_initial * (1.0 - (i + ((epoch-1) * len(train_loader))) / max_iterations) ** 0.9
            
            if max_mean_dice > 0.8:
                lr_ *= 0.1
            if max_mean_dice > 0.805:
                lr_ *= 0.1
            if max_mean_dice > 0.81:    
                lr_ *= 0.1

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            if loss < min_loss:
                min_loss = loss
                lr = optimizer.state_dict()['param_groups'][0]['lr']
                logstr = 'epoch: %d, iterator: %d, loss: %.5f, lr: %.8f, loss_dice: %.5f, loss_ce: %.5f' % (epoch, i, loss, lr, loss_dice, loss_ce)
                save.log(logstr)
            
        # evaluation
        logstr = 'epoch: %d, iterator: %d, loss: %.5f, lr: %.8f, loss_dice: %.5f, loss_ce: %.5f' % (epoch, i, loss, lr_, loss_dice, loss_ce)
        save.log(logstr)
    
        if (epoch < 50) or ( (epoch %5 != 0) and (epoch < 230)):
            continue

        net.eval()
        metric_list = 0.0
        for i, data in enumerate(val_loader, 0): 
            h, w = data["image"].size()[2:]
            image, label, case_name = data["image"], data["label"], data['case_name'][0]
            metric_i, outputs = test_single_volume(image, label, net, classes=args.num_classes, patch_size=[args.patch_size, args.patch_size],
                                      test_save_path=None, case=case_name, z_spacing=1)

            if max_mean_dice > 0.78:
                save.image(case_name, image[0], outputs, label[0])

            metric_list += np.array(metric_i)
            logstr = 'idx %d case %s mean_dice %f mean_hd95 %f' % (i, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1])
            save.log(logstr)
        metric_list = metric_list / len(val_loader)
        for i in range(1, args.num_classes):
            logstr = 'Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1])
            save.log(logstr)
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        logstr = 'Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95)
        save.log(logstr)

        if performance > max_mean_dice:
            max_mean_dice = performance
            fn = 'best_dice_epoch_%d_%.6f_%.6f.pth' % (epoch, performance, mean_hd95)
            save.net(net, fn)
            
            
    save("Training Finished !")

if __name__ == "__main__":
    main()
