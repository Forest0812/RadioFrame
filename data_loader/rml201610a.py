import random
import pickle
import os
import numpy as np
import torch
from torch.utils.data import Dataset


def load_data(dirname, prop):
    """[加载RML2016.10a数据集]

    Args:
        dirname ([str]]): [RML2016.10a数据集所在的绝对路径]
        prop ([float]]): [训练集所占的比例]
        expand_dims ([bool]): 是否需要拓展维度, 用户LSTM模型的输入

    Returns:
        [train_x]: [训练数据，(220000*prop,1,2,128)，若不拓展维度，输出为(220000*prop,2,128) 以下同理]
        [train_y]: [训练标签，(220000*prop,1,)]
        [test_x]: [测试数据，(220000*(1-prop),1,2,128)]
        [test_y]: [测试标签，(220000*(1-prop),1,)]
        [snrs]: [数据中所包含的所有的SNR,list[int]]
        [mods]: [数据中所包含的所有的调制类型,list[str]] 其index对应与标签
    """
    f = open(os.path.join(dirname, 'RML2016.10a_dict.pkl'), 'rb')
    rml_data = pickle.load(f, encoding = 'latin1')
    snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], rml_data.keys())))), [1,0])
    datas = []
    labels = []
    for mod in mods:
        for snr in snrs:
            datas.append(rml_data[(mod, snr)])
            for i in range(rml_data[(mod, snr)].shape[0]):
                labels.append((mod, snr))
    datas = np.vstack(datas)
    datas = np.expand_dims(datas, axis = 1)

    np.random.seed(2016)
    n_examples = datas.shape[0]
    n_train = int(n_examples * 0.5)
    train_idx = np.random.choice(range(0, n_examples),size = n_train,replace = False)
    test_idx = list(set(range(0,n_examples)) - set(train_idx))
    train_x = datas[train_idx]
    test_x = datas[test_idx]
    train_y = list(map(lambda x: mods.index(labels[x][0]), train_idx))
    train_snr = list(map(lambda x: labels[x][1], train_idx))
    test_y = list(map(lambda x: mods.index(labels[x][0]), test_idx))
    test_snr = list(map(lambda x: labels[x][1], test_idx))
    return train_x, train_y, train_snr, test_x, test_y, test_snr, snrs, mods

def data_save(dirname, prop):
    """[加载RML2016.10a数据集]
    将数据存储到文件中
    训练数据train_x, train_y, train_snr 存储到文件'train_data.p'中，加载方式如下：
        train_x, train_y, train_snr = pickle.load(open(os.path.join(dirname, 'train_data.p'), 'rb'))
        train_x : 训练数据，(220000*prop,1,2,128)，若不拓展维度，输出为(220000*prop,2,128) 以下同理
        train_y : 训练标签，(220000*prop,1,)
        train_snr : 训练数据所对应的snr , (220000*prop,1)
    测试数据test_x, test_y, test_snr 存储到文件'test_data.p'中，加载方式如下：
        test_x, test_y, test_snr = pickle.load(open(os.path.join(dirname, 'test_data.p'), 'rb'))
        test_x : 测试数据，(220000*(1-prop),1,2,128)
        test_y : 测试标签，(220000*(1-prop),1,)
        test_snr : 测试数据所对应的snr , (220000*(1-prop),1,)
    各类超参数snrs(SNR种类)，mods(调制方式种类) 存储到文件'augments.p'中，加载方式如下：
        snrs, mods = pickle.load(open(os.path.join(dirname, 'augments.p'), 'rb'))
        snrs : 数据中所包含的所有的SNR,list[int]
        mods : 数据中所包含的所有的调制类型,list[str] 其index对应与标签

    Args:
        dirname ([str]]): [RML2016.10a数据集所在的绝对路径]
        prop ([float]]): [训练集所占的比例]

    Returns:

    """
    train_x, train_y, train_snr, test_x, test_y, test_snr, snrs, mods = load_data(dirname, prop)
    pickle.dump([train_x, train_y, train_snr], open(os.path.join(dirname, 'train_data.p'), 'wb'))
    pickle.dump([test_x, test_y, test_snr], open(os.path.join(dirname, 'test_data.p'), 'wb'))
    pickle.dump([snrs, mods], open(os.path.join(dirname, 'augments.p'), 'wb'))



class Rml2016_10aTrainSet(Dataset):
    def __init__(self,dirname, prop):
        """[加载RML2016.10a训练集]

        Args:
            dirname ([str]): [RML2016.10a文件的绝对路径]]
            prop ([float]): [训练集所占的比例]
        """
        if not os.path.exists(os.path.join(dirname, 'train_data.p')):
            data_save(dirname, prop)     
            print('Data genreated Successfully')
        
        train_x, train_y, train_snr = pickle.load(open(os.path.join(dirname, 'train_data.p'), 'rb'))
        snrs, mods = pickle.load(open(os.path.join(dirname, 'augments.p'), 'rb'))
        self.data = train_x
        self.labels = train_y
        self.train_snr = train_snr
        self.snrs = snrs
        self.mods = mods
        
    def __getitem__(self,idx):
        return (self.data[idx],self.labels[idx], self.train_snr[idx])
    
    def __len__(self):
        return len(self.data)

    def get_snr_and_mod(self):
        return self.snrs, self.mods


class Rml2016_10aTestSet(Dataset):
    def __init__(self,dirname, prop):
        """[加载RML2016.10a训练集]

        Args:
            dirname ([str]): [RML2016.10a文件的绝对路径]]
            prop ([float]): [训练集所占的比例]
        """        

        test_x, test_y, test_snr = pickle.load(open(os.path.join(dirname, 'test_data.p'), 'rb'))
        snrs, mods = pickle.load(open(os.path.join(dirname, 'augments.p'), 'rb'))

        self.data = test_x
        self.labels = test_y
        self.data_snr = test_snr
        self.snrs = snrs
        self.mods = mods
        
    def __getitem__(self,idx):
        return (self.data[idx], self.labels[idx], self.data_snr[idx])
    
    def __len__(self):
        return len(self.data)

    def get_snr_and_mod(self):
        return self.snrs, self.mods