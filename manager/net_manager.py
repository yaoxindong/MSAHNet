from net.TransUNet import *
from net.DeepResUNet import DeepResUNet_AOT, DeepResUNet, DeepResUNet_RESAOT
from net.UNet import UNet
from net.MSAHNet import MSAHNet


def get_net(args):
    net_name = args.net
    netargsstrs = ''
    if net_name == 'UNet':
        net = UNet()
    elif net_name == 'DeepResUNet':
        net = DeepResUNet()
    elif net_name == 'TransUNet':
        config_vit = CONFIGS[args.vit_name]
        config_vit.n_classes = args.num_classes
        config_vit.n_skip = args.n_skip
        if args.vit_name.find('R50') != -1:
            config_vit.patches.grid = (int(args.patch_size / args.vit_patch_size), int(args.patch_size / args.vit_patch_size))
        net = VisionTransformer(config_vit, img_size=args.patch_size, num_classes=config_vit.n_classes)
    elif net_name == 'MSAHNet':
        config_vit = CONFIGS[args.vit_name]
        config_vit.n_classes = args.num_classes
        config_vit.n_skip = args.n_skip
        if args.vit_name.find('R50') != -1:
            config_vit.patches.grid = (int(args.patch_size / args.vit_patch_size), int(args.patch_size / args.vit_patch_size))
        net = MSAHNet(config_vit, img_size=args.patch_size, num_classes=config_vit.n_classes)

    else:
        raise Exception("model name error!")

    

    
    return net,netargsstrs