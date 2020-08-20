import torch 
from torch import nn 

class Distillation(object):
    
    """
    Distillation loss class 
    supports :param: method--> 
        1. mse      : match yolo layer outputs  
    """

    def __init__(self,method="mse"): 
        if method=="mse":
            self.loss_fn = self.mse
        else:
            raise NotImplementedError 

    def mse(self, predS, predT): 
        """
        mse between predT & predS
        only works when Stu & Tea are same architecture 
        """ 
        assert len(predT) == len(predS) 
        dLoss = []
        for branchS, branchT in zip(predS, predT):
            dLoss.append(torch.mean((branchS - branchT)**2))
        dLoss = sum(dLoss)
        return dLoss     
