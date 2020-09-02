import torch 
from torch import nn
import torch.nn.functional as nnfunc

class Distillation(object):
    
    """
    Distillation loss class 
    supports :param: method--> 
        1. mse      : match yolo layer outputs  
    """

    def __init__(self,method="mse"): 
        if method=="mse":
            self.loss_fn = self.mse
        elif method=="cfmse":
            self.loss_fn = self.cfmse
        elif method=="cfmse2":
            self.loss_fn = self.cfmse2
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
        dLoss_items = torch.tensor((0.0, 0.0, 0.0, dLoss.item())).to(dLoss.device)
        return dLoss, dLoss_items.detach()

    def cfmse(self, predS, predT):
        """
        mse between predT & predS weighted based on confidence
        only works when Stu & Tea are same architecture
        """
        assert len(predT) == len(predS)
        dLoss = []
        for branchS, branchT in zip(predS, predT):
            conf_sgm = torch.sigmoid(branchT[..., 4:5])
            conf_err = torch.mean( (branchS[..., 4] - branchT[..., 4])**2 )
            dims_err = torch.mean( (branchS[...,:4] - branchT[...,:4])**2 * conf_sgm )
            clss_err = torch.mean( (branchS[...,5:] - branchT[...,5:])**2 * conf_sgm )
            dLoss.append(conf_err + dims_err + clss_err)
        dLoss = sum(dLoss)
        return dLoss

    def cfmse2(self, predS, predT):
        """
        mse between predT & predS weighted based on confidence
        only works when Stu & Tea are same architecture
        """
        assert len(predT) == len(predS)
        conf_errs, dims_errs, clss_errs = [],[],[]
        for branchS, branchT in zip(predS, predT):

            branchS, branchT = branchS.view(-1, 85), branchT.view(-1, 85)

            teacher_conf = torch.sigmoid(branchT[..., 4])
            student_conf = torch.sigmoid(branchS[..., 4])
            conf_err     = nnfunc.mse_loss(student_conf, teacher_conf, reduction='mean')

            teacher_dims = branchT[..., :4]
            student_dims = branchS[..., :4]
            dims_err     = nnfunc.mse_loss(student_dims, teacher_dims, reduction='none').sum(dim=-1) * teacher_conf
            dims_err     = torch.mean(dims_err)

            tau = 3 # temperature
            teacher_clss = torch.argmax(branchT[..., 5:], dim=-1) # nnfunc.softmax(branchT[..., 5:]/tau, dim=-1)
            student_clss = branchS[..., 5:] # /tau
            # Torch KLDivLoss : input is expected to contain log-probs, target is expected to be probs
            # clss_err     = nnfunc.kl_div(student_clss, teacher_clss, reduction='none').sum(dim=-1) * teacher_conf
            # clss_err     = -1.0 * torch.mean(clss_err)
            clss_err     = nnfunc.cross_entropy(student_clss, teacher_clss, reduction='none') * teacher_conf
            clss_err     = torch.mean(clss_err)

            conf_errs.append(conf_err)
            dims_errs.append(dims_err)
            clss_errs.append(clss_err)

            # dLoss.append(1000.0*conf_err + 100*dims_err + 100*clss_err)
        conf_errs, dims_errs, clss_errs = sum(conf_errs), sum(dims_errs), sum(clss_errs)
        dLoss = 1000*conf_errs + 100*dims_errs + 100*clss_errs
        dLoss_items = torch.tensor((conf_errs, dims_errs, clss_errs, dLoss.item())).to(dLoss.device)

        return dLoss, dLoss_items.detach()