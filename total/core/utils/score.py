import numpy as np
def prep_clf(obs, pre, threshold):
    '''
    func: 计算二分类结果-混淆矩阵的四个元素
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。

    returns:
        hits, misses, falsealarms, correctnegatives
        #aliases: TP, FN, FP, TN
    '''
    # 根据阈值分类为 0, 1
    obs = np.where(obs >= threshold, 1, 0)
    pre = np.where(pre >= threshold, 1, 0)

    # True positive (TP)
    hits = np.sum((obs == 1) & (pre == 1))

    # False negative (FN)
    misses = np.sum((obs == 1) & (pre == 0))

    # False positive (FP)
    falsealarms = np.sum((obs == 0) & (pre == 1))

    # True negative (TN)
    correctnegatives = np.sum((obs == 0) & (pre == 0))

    return hits.astype(np.float64), misses.astype(np.float64), falsealarms.astype(np.float64), correctnegatives.astype(np.float64)


def precision(obs, pre, threshold):
    '''
    func: 计算精确度precision: TP / (TP + FP)
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。

    returns:
        dtype: float
    '''

    TP, FN, FP, TN = prep_clf(obs=obs, pre=pre, threshold=threshold)

    return TP / (TP + FP)


def recall(obs, pre, threshold):
    '''
    func: 计算召回率recall: TP / (TP + FN)
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。

    returns:
        dtype: float
    '''

    TP, FN, FP, TN = prep_clf(obs=obs, pre=pre, threshold=threshold)

    return TP / (TP + FN)


def ACC(obs, pre, threshold=0.1):
    '''
    func: 计算准确度Accuracy: (TP + TN) / (TP + TN + FP + FN)
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。

    returns:
        dtype: float
    '''

    TP, FN, FP, TN = prep_clf(obs=obs, pre=pre, threshold=threshold)

    return (TP + TN) / (TP + TN + FP + FN)


def FSC(obs, pre, threshold):
    '''
    func:计算f1 score = 2 * ((precision * recall) / (precision + recall))
    '''
    precision_socre = precision(obs, pre, threshold=threshold)
    recall_score = recall(obs, pre, threshold=threshold)

    return 2 * ((precision_socre * recall_score) / (precision_socre + recall_score))
def FAR(obs, pre, threshold):
    '''
    func: 计算误警率。falsealarms / (hits + falsealarms)
    FAR - false alarm rate
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): prediction
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)
    Returns:
        float: FAR value
    '''
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre = pre,
                                                           threshold=threshold)

    return falsealarms / (hits + falsealarms)

def MAR(obs, pre, threshold):
    '''
    func : 计算漏报率 misses / (hits + misses)
    MAR - Missing Alarm Rate
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): prediction
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)
    Returns:
        float: MAR value
    '''
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre = pre,
                                                           threshold=threshold)

    return misses / (hits + misses)
def POD(obs, pre, threshold):
    '''
    func : 计算命中率 hits / (hits + misses)
    pod - Probability of Detection
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): prediction
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)
    Returns:
        float: PDO value
    '''
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre = pre,threshold=threshold)

    if (hits + misses) == 0:
      return 0
    else:
         return hits / (hits + misses)
    #
    #
    # return hits / (hits + misses)

# def HSS(obs, pre, threshold):
#     '''
#     HSS - Heidke skill score
#     Args:
#         obs (numpy.ndarray): observations
#         pre (numpy.ndarray): pre
#         threshold (float)  : threshold for rainfall values binaryzation
#                              (rain/no rain)
#     Returns:
#         float: HSS value
#     '''
#     hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre = pre,threshold=threshold)
#     ts=hits * correctnegatives
#     hs=misses * falsealarms
#     HSS_num = 2 * (ts - hs)
#     s=misses**2
#     h=falsealarms**2
#     d=(misses + falsealarms)*(hits + correctnegatives)
#
#     HSS_den = (s + h + 2*hits*correctnegatives +d)
#
#     if (HSS_den)==0:
#         return 0
#     else:
#         return HSS_num / HSS_den
def HSS(obs, pre, threshold):
    '''
    HSS - Heidke skill score
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): pre
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)
    Returns:
        float: HSS value
    '''
    TP, FN, FP, TN = prep_clf(obs=obs, pre = pre, threshold=threshold)


    HSS_num = 2 * (TP*TN-FN*FP)
    HSS_den = (TP+FN)*(FN+TN)+(TP+FP)*(FP+TN)
    if (HSS_den)==0:
        return 0
    else:
        return HSS_num / HSS_den



def CSI(obs, pre, threshold):
    '''
    func: 计算TS评分: TS = hits/(hits + falsealarms + misses)
    	  alias: TP/(TP+FP+FN)
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
    returns:
        dtype: float
    '''

    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre=pre, threshold=threshold)

    if (hits + falsealarms + misses)==0:
        return 0
    else:
        return hits / (hits + falsealarms + misses)
