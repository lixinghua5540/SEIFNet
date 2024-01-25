import os
import numpy as np
import matplotlib.pyplot as plt

from models.networks import *
from misc.metric_tool import ConfuseMatrixMeter
from misc.logger_tool import Logger
from utils_ import de_norm
import utils_
from tqdm import tqdm

# Decide which device we want to run on
# torch.cuda.current_device()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CDEvaluator():

    def __init__(self, args, dataloader):

        self.dataloader = dataloader

        self.n_class = args.n_class
        # define G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)
        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids)>0
                                   else "cpu")
        print(self.device)

        # define some other vars to record the training states
        self.running_metric = ConfuseMatrixMeter(n_class=self.n_class)

        # define logger file
        logger_path = os.path.join(args.checkpoint_dir, 'log_test.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)


        #  training log
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0

        self.steps_per_epoch = len(dataloader)

        self.G_pred = None
        self.pred_vis = None
        self.batch = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)


    def _load_checkpoint(self, checkpoint_name='best_ckpt.pt'):

        if os.path.exists(os.path.join(self.checkpoint_dir, checkpoint_name)):
            self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, checkpoint_name), map_location=self.device)

            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])

            self.net_G.to(self.device)

            # update some other states
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.logger.write('Eval Historical_best_acc = %.4f (at epoch %d)\n' %
                  (self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')

        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)


    def _visualize_pred(self,args):
        # pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        if args.deep_supervision== True:
            pred =torch.argmax(self.G_pred[-1], dim=1, keepdim=True)
        else:
            pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis


    def _update_metric(self,args):
        """
        update metric
        """
        target = self.batch['L'].to(self.device).detach()
        # G_pred = self.G_pred.detach()
        if args.deep_supervision == True:
            G_pred = self.G_pred[-1].detach()
        else:
            G_pred = self.G_pred.detach()
        G_pred = torch.argmax(G_pred, dim=1)

        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
        return current_score

    def _collect_running_batch_states(self,args):

        running_acc = self._update_metric(args)#变化的精确度

        m = len(self.dataloader)
        #print('m:',m)
        #print('batch_id:',self.batch_id)

        if np.mod(self.batch_id, 100) == 1:#取模运算，同正为正，同负为负
            message = 'Is_training: %s. [%d,%d],  running_mf1: %.5f\n' %\
                      (self.is_training, self.batch_id, m, running_acc)
            self.logger.write(message)

        if np.mod(self.batch_id, 100) == 1:
            vis_input = utils_.make_numpy_grid(de_norm(self.batch['A']))
            vis_input2 = utils_.make_numpy_grid(de_norm(self.batch['B']))

            vis_pred = utils_.make_numpy_grid(self._visualize_pred(args))

            vis_gt = utils_.make_numpy_grid(self.batch['L'])
            vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
            vis = np.clip(vis, a_min=0.0, a_max=1.0)#限制数组中的值，也就是说clip这个函数将数组中的元素限制在a_min, a_max之间，大于a_max的就使得它等于 a_max，小于a_min,的就使得它等于a_min
            file_name = os.path.join(
                self.vis_dir, 'eval_' + str(self.batch_id)+'.jpg')
            plt.imsave(file_name, vis)



    def _collect_epoch_states(self):

        scores_dict = self.running_metric.get_scores()

        np.save(os.path.join(self.checkpoint_dir, 'scores_dict.npy'), scores_dict)

        self.epoch_acc = scores_dict['mf1']
        print(self.epoch_acc )

        with open(os.path.join(self.checkpoint_dir, '%s.txt' % (self.epoch_acc)),
                  mode='a') as file:
            pass

        message = ''
        for k, v in scores_dict.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write('%s\n' % message)  # save the message

        self.logger.write('\n')

    def _clear_cache(self):
        self.running_metric.clear()

    def  _forward_pass(self,batch,args):
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)
        # sobel = batch['S'].to(self.device)
        # self.G_pred1, self.G_pred2, self.G_pred3 = self.net_G(img_in1, img_in2)
        # self.G_pred = self.G_pred1 + self.G_pred2 + self.G_pred3
        if args.loss_SD== True :
            # self.G_pred1, self.G_pred2, self.G_pred3 = self.net_G(img_in1, img_in2,sobel)
            # self.G_pred = self.G_pred1 + self.G_pred2 + self.G_pred3
            #cd_net
            # self.G_pred0, self.G_pred1, self.G_pred2, self.G_pred3, self.G_pred4 = self.net_G(img_in1, img_in2)
            # self.G_pred = self.G_pred0

            #DMINet
            self.G_pred0, self.G_pred1, self.G_pred2, self.G_pred3 = self.net_G(img_in1, img_in2)
            self.G_pred = self.G_pred0 + self.G_pred1

        else :
            # self.G_pred = self.net_G(img_in1, img_in2,sobel)
            self.G_pred = self.net_G(img_in1, img_in2)

    def eval_models(self,args,checkpoint_name='best_ckpt.pt'):

        self._load_checkpoint(checkpoint_name)

        ################## Eval ##################
        ##########################################
        self.logger.write('Begin evaluation...\n')
        self._clear_cache()#不返回任何值，清空running_metric
        self.is_training = False
        self.net_G.eval()

        #Iterate over data.遍历数据（原始）
        for self.batch_id, batch in enumerate(self.dataloader, 0):
            #name = batch['name']
            #print('process: %s' % name)
            with torch.no_grad():
                self._forward_pass(batch,args)
            self._collect_running_batch_states(args)
        self._collect_epoch_states()


