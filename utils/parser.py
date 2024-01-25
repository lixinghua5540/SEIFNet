import argparse as ag
import json

def get_parser_with_args(metadata_json='metadata.json'):
    parser = ag.ArgumentParser(description='Training change detection network')

    with open(metadata_json, 'r') as fin:
        metadata = json.load(fin)
        parser.set_defaults(**metadata)

    #project save
    parser.add_argument('--project name', default='LEVIR+_ViTAE_BIT_bce_100', type=str)
    parser.add_argument('--path', default='checkpoints', type=str, help='path of saved model')
    # parser.add_argument('--checkpoint_root', default='checkpoints', type=str)

    #network
    parser.add_argument('--backbone', default='vitae', type=str, choices=['resnet','swin','vitae'], help='type of model')

    parser.add_argument('--dataset', default='levir+', type=str, choices=['cdd','levir','levir+'], help='type of dataset')

    parser.add_argument('--mode', default='rsp_100', type=str, choices=['imp','rsp_40', 'rsp_100', 'rsp_120' , 'rsp_300', 'rsp_300_sgd', 'seco'], help='type of pretrn')


    return parser, metadata

