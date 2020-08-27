import numpy as np



def accuary(logits,target):
    """[计算准确率]

    Args:
        logits ([array,matrix]): [网络模型输出]
        target ([array,int]): [目标]

    Returns:
        [type]: [description]
    """
    acc = (logits.argmax(1) == target).sum()
    acc = acc/target.shape[0]
    return acc