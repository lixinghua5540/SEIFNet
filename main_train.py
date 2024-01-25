from argparse import ArgumentParser
import torch
from models.trainer import *
from utils_ import str2bool
print(torch.cuda.is_available())

"""
the main function for training the CD networks
"""


def train(args):
    dataloaders = utils_.get_loaders(args)
    model = CDTrainer(args=args, dataloaders=dataloaders)
    model.train_models(args=args)
    # model.train_models()


def test(args):
    from models.evaluator import CDEvaluator
    dataloader = utils_.get_loader(args.data_name, img_size=args.img_size,
                                  batch_size=args.batch_size, is_train=False,
                                  split='test')
    model = CDEvaluator(args=args, dataloader=dataloader)

    model.eval_models(args)


if __name__ == '__main__':
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--project_name', default='LEVIR-CD_SEIFNet_ce_Adamw_0.0001_200', type=str)
    #SYSU_res18_coMDE2_AFF_Bcedice_l0_Adamw_0.0001_200
    #LEVIR_transformer_CoDEM_AFF_ce_Adamw_0.0001_200_2
    parser.add_argument('--checkpoint_root', default='checkpoints', type=str)

    # data
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--data_name', default='LEVIR', type=str,help='ChangeDetection|LEVIR|DSFIN|SYSU-CD|LEVIR+|BBCD|WHU-CD')

    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--split', default="train", type=str)
    parser.add_argument('--split_val', default="val", type=str)

    parser.add_argument('--img_size', default=256, type=int)

    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--embed_dim', default=64, type=int)
    parser.add_argument('--net_G', default='SEIFNet', type=str,
                        help='FC_EF | FC_Siam_conc | '
                             'FC_Siam_diff | UNet++|SNUNet|'
                             'DTCDSCN|IFNet|'
                             'base_transformer_pos_s4_dd8_dedim8|'
                             'ChangeFormer|'
                             'A2Net|DMINet|TFI-GR|'
                            'SEIFNet')
    parser.add_argument('--backbone', default='L-Backbone', type=str, choices=['resnet', 'swin', 'vitae','L-Backbone-cross','L-Backbone','BiFormer'],
                        help='type of model')
    parser.add_argument('--mode', default='None', type=str,
                        choices=['imp','res18 ','rsp_40', 'rsp_100', 'rsp_120', 'rsp_300', 'rsp_300_sgd', 'seco','None'],
                        help='type of pretrn')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)#UNet++和A2net时为True，不需要时为False

    parser.add_argument('--loss_SD', default=False,type=str2bool) #IFNet DMINet才为True
    # optimizer
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--max_epochs', default=200, type=int) #150
    parser.add_argument('--lr_policy', default='linear', type=str,
                        help='linear | step')
    parser.add_argument('--lr_decay_iters', default=200, type=int)

    args = parser.parse_args()
    utils_.get_device(args)
    print(args.gpu_ids)

    #  checkpoints dir
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    #  visualize dir
    args.vis_dir = os.path.join('vis', args.project_name)
    os.makedirs(args.vis_dir, exist_ok=True)

    train(args)

    test(args)
