import argparse

class Parameter():
    
    def __init__(self):
        self.args  = self.init_()

    def init_(self):    

        parser = argparse.ArgumentParser()
        # Common settings
        parser.add_argument('--batch_size', type=int, default=8, help='batch size')
        parser.add_argument('--epoch', type=int, default=250, help='training epochs')
        parser.add_argument('--gpu', type=str, default='3', help='GPUs')
        parser.add_argument('--lr_initial', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
        parser.add_argument('--num_classes', type=int, default ='9', help='output channel of network')

        # training settings
        parser.add_argument('--patch_size', type=int, default=224, help='patch size of training sample')
        parser.add_argument('--data_path', type=str, default ='/media/yang/Pytorch/yao/dataset',  help=' data path')

        parser.add_argument('--dataset', type=str, default ='Synapse')
        parser.add_argument('--train_workers', type=int, default=16, help='train_dataloader workers')
        parser.add_argument('--eval_workers', type=int, default=8, help='eval_dataloader workers')
        parser.add_argument('--net', type=str, default ='TransUNet_DANet_U',  help='archtechture') #TransUNet_DANet_U TransUNet
        
        # args for vit
        parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')
        parser.add_argument('--vit_patch_size', type=int, default=16, help='vit patch_size')

        
        parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')

        return parser
