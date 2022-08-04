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

    path1 = '../save/2021-12-30_10-55-21(TransUNet_DANet_U)/'

    net = torch.load('../save/2021-12-30_10-55-21(TransUNet_DANet_U)/*')


    ######### Set GPUs ###########
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu 

    net.cuda()
    ######### DataLoader ###########

    val_dataset = manager.get_validation_data(args.dataset, args.data_path)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, 
                            num_workers=1, pin_memory=False, drop_last=False)    

    torch.cuda.empty_cache()
    net.eval()
    metric_list = 0.0
    for i, data in enumerate(val_loader, 0):
    ######## test ########
        image, label, case_name = data["image"], data["label"], data['case_name'][0]
        metric_i, outputs = test_single_volume(image, label, net, classes=args.num_classes, patch_size=[args.patch_size, args.patch_size],
                                      test_save_path=None, case=case_name, z_spacing=1)

        # save.image(case_name, image[0], outputs, label[0])
        metric_list += np.array(metric_i)
        logstr = 'idx %d case %s mean_dice %f mean_hd95 %f' % (i, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1])
        print(logstr)
        # save.log(logstr)
    metric_list = metric_list / len(val_loader)
    for i in range(1, args.num_classes):
        logstr = 'Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1])
        print(logstr)
        # save.log(logstr)
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logstr = 'Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95)
    # save.log(logstr)
    print(logstr)

if __name__ == "__main__":
    main()
