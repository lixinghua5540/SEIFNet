from argparse import ArgumentParser
import torch
from models.evaluator import *
from utils_ import str2bool
print(torch.cuda.is_available())


"""
eval the CD model
"""

def main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--project_name', default='LEVIR_SEIFNet_ce_Adamw_0.0001_150', type=str)
    parser.add_argument('--print_models', default=False, type=bool, help='print models')
    parser.add_argument('--checkpoints_root', default='checkpoints', type=str)
    parser.add_argument('--vis_root', default='vis', type=str)

    # data
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--data_name', default='LEVIR', type=str)

    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--split', default="test", type=str)

    parser.add_argument('--img_size', default=256, type=int)

    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--embed_dim', default=64, type=int)
    parser.add_argument('--net_G', default='SEIFNet', type=str,
                        help='base_resnet18 | base_transformer_pos_s4_dd8 | base_transformer_pos_s4_dd8_dedim8|'
                             'vitae_transformer|SEIFNet')
    parser.add_argument('--backbone', default='L-Backbone', type=str, choices=['resnet', 'swin', 'vitae'],
                        help='type of model')
    parser.add_argument('--mode', default='None', type=str,
                        choices=['imp', 'rsp_40', 'rsp_100', 'rsp_120', 'rsp_300', 'rsp_300_sgd', 'seco'],
                        help='type of pretrn')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)  # UNet++和A2net时为True，不需要时为False
    # parser.add_argument('--loss1', default='Focal', type=str,help='ce|BL_ce|Focal|Focal_Dice|Focal_Dice_BL|Focal_BL|BL_Focal|Focal_BF_IOU')
    # parser.add_argument('--loss2', default='BL_Focal', type=str,
    #                     help='ce|BL_ce|Focal|Focal_Dice|Focal_Dice_BL|Focal_BL|BL_Focal|Focal_BF_IOU')
    parser.add_argument('--loss_SD', default=True, type=str2bool)  # 只有CD_Net才为True

    parser.add_argument('--checkpoint_name', default='best_ckpt.pt', type=str)

    args = parser.parse_args()
    utils_.get_device(args)
    print(args.gpu_ids)

    #  checkpoints dir
    args.checkpoint_dir = os.path.join(args.checkpoints_root, args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    #  visualize dir
    args.vis_dir = os.path.join(args.vis_root, args.project_name)
    os.makedirs(args.vis_dir, exist_ok=True)

    dataloader = utils_.get_loader(args.data_name, img_size=args.img_size,
                                  batch_size=args.batch_size, is_train=False,
                                  split=args.split)
    model = CDEvaluator(args=args, dataloader=dataloader)

    model.eval_models(args=args,checkpoint_name=args.checkpoint_name)


if __name__ == '__main__':
    main()

