import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import scipy.stats as st


'''每个输出的第 0 维（即第一个维度）是批量大小，第 1 维是序列长度，第 2 维是特征维度。
[:,1:,:] 表示去除每个输出的第一个序列元素（通常是分类令牌），保留其余序列元素及其特征。
dim=2 表示沿着特征维度进行拼接。'''
class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []


