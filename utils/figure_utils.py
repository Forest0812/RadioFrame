import numpy as np
import os
import matplotlib.pyplot as plt



def plot_confusion_matrix(cm, dirname, labels, title='Confusion matrix', cmap=plt.cm.Blues):
    """[给定混淆矩阵，绘制并保存]

    Args:
        cm ([二维array]): [混淆矩阵]
        dirname ([str]): [混淆矩阵图要存储的位置]
        labels (list, optional): [混淆矩阵的标签]
        title (str, optional): [description]. Defaults to 'Confusion matrix'.
        cmap ([type], optional): [混淆矩阵图中的颜色]. Defaults to plt.cm.Blues.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(dirname,title+'.jpg'))


def generate_confusion_matrix(predict, targets, classes):
    """[生成混淆矩阵]

    Args:
        predict ([二维array,(length, probality)]]): [网络的得到预测值]]
        targets ([一维array 或 二维array（onehot）]): [对应的真实标签]
        classes ([一维array]): 真实类别，str
    """
    one_dim = len(targets.shape) == 1
    conf = np.zeros([len(classes),len(classes)])
    confnorm = np.zeros([len(classes),len(classes)])
    for i in range(predict.shape[0]):
        if one_dim:
            j = targets[i]
        else:
            j = list(targets[i,:]).index(1)
        k = np.argmax(predict[i,:])
        conf[j][k] = conf[j][k] + 1
    for i in range(0,len(classes)):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
    return confnorm