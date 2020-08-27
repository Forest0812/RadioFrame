import torch
import torch.nn as nn
import os
import numpy as np
from config import *
from utils import *

class BaseTrainer(object):
    """[Base class for all trainers]

    Args:
        self.model ([]): 要训练的模型
        self.data_loader ([DataLoader]): 数据加载器
        self.criterion ([nn.loss]): 损失函数
        self.optimizer ([]): 优化器
        self.config ([]): 配置类
        self.checkpoint_dir ([str]): checkpoint文件夹
        self.checkpoint_filename ([str]): checkpoint's filename
        self.model_best_file ([str]): best checkpoint's filename
        self.log_filename ([str]): log's filename
        self.save_period ([int]): 存储checkpoint的周期
        self.EPOCH ([int]): 训练的最大epoch
        self.len_epoch ([int]): data_loader的batch数目
    """
    def __init__(self,model,data_loader,criterion,optimizer,metrics,config):
        self.model = model
        self.data_loader = data_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics = metrics
        self.config = config
        self.model.initialize_weight()
        #------------------------------------GPU配置
        self.use_gpu = False
        self.device_ids = [0]
        if self.config.GPU['use_gpu']:
            if not torch.cuda.is_available():
                print("There's no GPU is available , Now Automatically converted to CPU device")
            else:
                message = "There's no GPU is available"
                self.device_ids = self.config.GPU['device_id']
                assert len(self.device_ids) > 0,message
                self.model = self.model.cuda(self.device_ids[0])
                if len(self.device_ids) > 1:
                    self.model = nn.DataParallel(model, device_ids=self.device_ids)
                self.use_gpu = True

        #------------------------------------checkpoint配置
        self.checkpoint_dir = self.config.Checkpoint['checkpoint_dir']
        self.checkpoint_filename = os.path.join(self.checkpoint_dir, self.config.Checkpoint['checkpoint_file_format'])
        self.model_best_file = os.path.join(self.checkpoint_dir, self.config.Checkpoint['model_best'])
        self.log_filename = os.path.join(self.checkpoint_dir, self.config.Checkpoint['log_file'])
        self.save_period = self.config.Checkpoint['save_period']
        
        #------------------------------------load checkpoint
        if self.config.CONFIG['load_model']:
            model_filename = self.config.LoadModel['filename']          #test时加载的模型位置
            self._load_checkpoint(model_filename)

        #------------------------------------figure配置
        self.figure_dir = os.path.join('figure', self.config.CONFIG['model_name'] )
        
        #------------------------------------训练配置
        self.EPOCH = self.config.ARG['epoch']
        self.len_epoch = len(self.data_loader)

    
    def _train_epoch(self,epoch):
        """[Training logic for an epoch]

        Args:
            epoch ([int]): [目前的epoch number]
        """
        raise NotImplementedError

    def train(self):
        """[完整的训练逻辑]
        """
        Log = self.config.log_output()
        log_file = self.file_open(self.checkpoint_dir,self.log_filename)
        log = {'config':Log}
        self.file_write(log_file,log)

        self.model.train()
        for epoch in range(self.EPOCH):
            result = self._train_epoch(epoch)
            # optimizer adjust lr
            if self.config.CONFIG['adjust_lr']:
                self.optimizer = adjust_learning_rate(self.optimizer,epoch=epoch,**self.config.LrAdjust)

            # save logged informations into log dict
            log = {'epoch':epoch}
            for key, value in result.items():
                if key == 'metrics':
                    log.update({mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                else:
                    log[key] = value

            self.file_write(log_file,log)

            for key, value in log.items():
                print('    {:15s}: {}'.format(str(key), value))

                
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)


    def test(self):
        """[完整的测试逻辑]
        """
        Log = self.config.log_output()
        log_file = self.file_open(self.checkpoint_dir,self.log_filename)

        self.model.eval()
        result = self._test_epoch()
        log = {'config':Log}
        for key, value in result.items():
            if key == 'metrics':
                log.update({mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
            else:
                log[key] = value
        for key, value in log.items():
            print('    {:15s}: {}'.format(str(key), value))


    def _test_epoch(self):
        """[Testing logic for an epoch]
        """
        raise NotImplementedError


    def _eval_metrics(self,logits,targets):
        """[多种metric的运算]

        Args:
            logits ([array]): [网络模型输出]
            targets ([array]): [标签值]

        Returns:
            acc_metrics [array]: [多个metric对应的结果]
        """
        acc_metrics = np.zeros(len(self.metrics))
        for i,metric in enumerate(self.metrics):
            acc_metrics[i] = metric(logits,targets)
        return acc_metrics



    def file_open(self,dir,filename):
        ensure_dir(dir)
        f = open(filename,'w')
        return f

    def file_write(self,file,log):
        string = ''
        for key, value in log.items():
            string += key + '\t:'+str(value) +'\n'
        string += '\n\n\n'
        file.write(string)


    def _save_checkpoint(self, epoch):
        """[saving checkpoints]

        Args:
            epoch ([int]): [目前的epoch number]
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config
        }
        filename = self.checkpoint_filename.format(epoch)
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))


    def _load_checkpoint(self,model_filename):
        """[loading checkpoints]

        Args:
            model_filename ([str]): 预训练模型的文件名
        """
        message = "There's not checkpoint"
        assert os.path.exists(model_filename),message
        print("Loading checkpoint: {} ...".format(model_filename))
        checkpoint = torch.load(model_filename)
        self.model.load_state_dict(checkpoint['state_dict'])



            