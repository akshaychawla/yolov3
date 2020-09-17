import torch 
from torch import nn
import torch.nn.functional as nnfunc
from torchvision import utils as vutils
import torchvision.transforms.functional as vfunc
from PIL import Image,ImageDraw
from .utils import wh_iou, bbox_iou, smooth_BCE

def draw_boxes(imgs, targets, obj_conf, cls_score):
    """
    Draw targets on top of the images. labelled with obj_conf and cls_score
    Useful to see how boxes are filtered
    """
    imgs = imgs.detach().clone().cpu()
    pilimgs = []
    for idx in range(imgs.shape[0]):

        pilim = vfunc.to_pil_image(imgs[idx])
        width, height = pilim.size

        batchmask = targets[:,0] == idx

        im_targets  = targets[batchmask]
        im_obj_conf = obj_conf[batchmask]
        im_cls_score= cls_score[batchmask]

        for box,obj,cls in zip(im_targets, im_obj_conf, im_cls_score):

            x1 = (box[2] - (box[4]/2.0)) * width
            x2 = (box[2] + (box[4]/2.0)) * width
            y1 = (box[3] - (box[5]/2.0)) * height
            y2 = (box[3] + (box[5]/2.0)) * height

            pil_box = [x1, y1, x2, y2]
            pil_box = [int(atom) for atom in pil_box]

            draw = ImageDraw.Draw(pilim)
            maxwidth  = 20
            linewidth = int( max(1.0, float(obj)*maxwidth) )
            draw.rectangle(pil_box, outline=(250,0,0), width=linewidth)
            draw.text(pil_box[:2], "oconf:{:.3f} cconf:{:.3f}".format(float(obj), float(cls)), fill=(255,255,255))

        pilimgs.append(pilim)

    image_tensor = torch.stack( [ vfunc.to_tensor(atom) for atom in pilimgs ] , dim=0 )
    return image_tensor

def filter_targets(targets, obj_conf, cls_score, obj_thresh=0.0, cls_thresh=0.0):
    """
    Filter (targets, obj_conf, cls_score) on obj_thresh and cls_thresh)
    """
    mask = (obj_conf >= obj_thresh) * (cls_score >= cls_thresh) # AND operation
    targets_filtered    = targets[mask]
    obj_conf_filtered   = obj_conf[mask]
    cls_score_filtered  = cls_score[mask]
    return targets_filtered, obj_conf_filtered, cls_score_filtered

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
        elif method=="osd":
            self.loss_fn = self.osd
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

    def cfmse2(self, predS, predT, model):
        """
        mse between predT & predS weighted based on confidence
        only works when Stu & Tea are same architecture
        """
        import pdb; pdb.set_trace()
        assert len(predT) == len(predS)
        yolo_layers = [model.module_list[idx] for idx in model.yolo_layers]
        conf_errs, dims_errs, clss_errs = [],[],[]
        for branchS, branchT, yolo_layer in zip(predS, predT, yolo_layers):

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

    def osd(self, predS, infT, imgs, model):
        """
        objectness scaled distillation code
        """
        batch_size, _, height, width = imgs.shape
        targets, obj_conf, cls_score = self._toGT(infT, (height, width))

        # Filter boxes
        targets, obj_conf, cls_score = filter_targets(targets, obj_conf, cls_score, obj_thresh=0.1, cls_thresh=0.0)

        # Draw boxes
        # imgs_with_boxes = draw_boxes(imgs, targets, obj_conf, cls_score)
        # vutils.save_image(imgs_with_boxes, "outfile.png")

        # tack on the confidence score along with the targets
        targets = torch.cat((targets, obj_conf.view(-1,1)), dim=1)

        # Compute distillation loss
        loss, loss_items = compute_loss_distill(predS, targets, model)
        # loss *= batch_size / 64

        return loss, loss_items.detach()


    def _toGT(self, inf, img_size):
        """
        Convert inference output (inf) to GT format
        infT is of shape (bs x #boxes x 85) where each box is (center x, center y, w, h, obj, cls1, cls2 .. cls80)
            where center x, center y, w and h are pixels
        GT format: targets: (#boxes x 6) where each box is (batch_idx, class, x, y, w, h)
            where x, y, w, h are normalized to [0,1]
        img_size is tuple of size (height, width)
        """
        assert len(inf.shape) == 3
        assert len(img_size)  == 2
        height, width = img_size

        gt   = inf.clone().detach().view(-1, 85) # ( bs*boxes X 85 )
        cls_scores, cls_indices  = torch.max(gt[:,5:], dim=1, keepdim=True) # class prediction
        cls_indices = cls_indices.type(gt.dtype)
        obj  = gt[:,4] # objectness
        gt   = gt[:, :4]
        gt   = torch.cat((cls_indices, gt), dim=1)  # (cls,x,y,w,h)

        batchIdcs = []
        numBoxesPerImg = inf.shape[1]
        for batchIdx in range(inf.shape[0]):
            batchIdcs.extend([batchIdx] * numBoxesPerImg)
        assert len(batchIdcs) == inf.shape[0]*inf.shape[1]
        batchIdcs = torch.tensor(batchIdcs, dtype=gt.dtype, device=gt.device)
        batchIdcs = batchIdcs.unsqueeze(1)

        gt   = torch.cat((batchIdcs, gt), dim=1) # (batch,cls,x,y,w,h)

        # normalize to [0,1]
        gt[:, 2] /= width
        gt[:, 4] /= width
        gt[:, 3] /= height
        gt[:, 5] /= height

        return gt, obj, cls_scores.squeeze(1)

def build_targets_distill(p, targets, model):
    # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    nt = targets.shape[0]
    tcls, tbox, indices, anch, tobjscores = [], [], [], [], []
    gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
    off = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device).float()  # overlap offsets

    style = None
    multi_gpu = type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    for i, j in enumerate(model.yolo_layers):
        anchors = model.module.module_list[j].anchor_vec if multi_gpu else model.module_list[j].anchor_vec
        gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
        na = anchors.shape[0]  # number of anchors
        at = torch.arange(na).view(na, 1).repeat(1, nt)  # anchor tensor, same as .repeat_interleave(nt)

        # Match targets to anchors
        a, t, offsets = [], targets * gain, 0
        if nt:
            # r = t[None, :, 4:6] / anchors[:, None]  # wh ratio
            # j = torch.max(r, 1. / r).max(2)[0] < model.hyp['anchor_t']  # compare
            j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n) = wh_iou(anchors(3,2), gwh(n,2))
            a, t = at[j], t.repeat(na, 1, 1)[j]  # filter

            # overlaps
            gxy = t[:, 2:4]  # grid xy
            z = torch.zeros_like(gxy)
            if style == 'rect2':
                g = 0.2  # offset
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                a, t = torch.cat((a, a[j], a[k]), 0), torch.cat((t, t[j], t[k]), 0)
                offsets = torch.cat((z, z[j] + off[0], z[k] + off[1]), 0) * g

            elif style == 'rect4':
                g = 0.5  # offset
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxy % 1. > (1 - g)) & (gxy < (gain[[2, 3]] - 1.))).T
                a, t = torch.cat((a, a[j], a[k], a[l], a[m]), 0), torch.cat((t, t[j], t[k], t[l], t[m]), 0)
                offsets = torch.cat((z, z[j] + off[0], z[k] + off[1], z[l] + off[2], z[m] + off[3]), 0) * g

        # Define
        b, c = t[:, :2].long().T  # image, class
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        gij = (gxy - offsets).long()
        gi, gj = gij.T  # grid xy indices

        # Append
        indices.append((b, a, gj, gi))  # image, anchor, grid indices
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        anch.append(anchors[a])  # anchors
        tcls.append(c)  # class
        tobjscores.append(t[:,6])
        if c.shape[0]:  # if any targets
            assert c.max() < model.nc, 'Model accepts %g classes labeled from 0-%g, however you labelled a class %g. ' \
                                       'See https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data' % (
                                           model.nc, model.nc - 1, c.max())

    return tcls, tbox, indices, anch, tobjscores

def compute_loss_distill(p, targets, model):  # predictions, targets, model
    ft = torch.cuda.FloatTensor if p[0].is_cuda else torch.Tensor
    lcls, lbox, lobj = ft([0]), ft([0]), ft([0])
    tcls, tbox, indices, anchors, tobjscores = build_targets_distill(p, targets, model)  # targets
    h = model.hyp  # hyperparameters
    red = 'mean'  # Loss reduction (sum or mean)

    # Define criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=ft([h['cls_pw']]), reduction="none")
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=ft([h['obj_pw']]), reduction=red)

    # class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
    cp, cn = smooth_BCE(eps=0.0)

    # focal loss
    g = h['fl_gamma']  # focal loss gamma
    if g > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

    # per output
    nt = 0  # targets
    for i, pi in enumerate(p):  # layer index, layer predictions
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        tobj = torch.zeros_like(pi[..., 0])  # target obj

        nb = b.shape[0]  # number of targets
        if nb:
            nt += nb  # cumulative targets
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

            # GIoU
            pxy = ps[:, :2].sigmoid()
            pwh = ps[:, 2:4].exp().clamp(max=1E3) * anchors[i]
            pbox = torch.cat((pxy, pwh), 1)  # predicted box
            giou = bbox_iou(pbox.t(), tbox[i], x1y1x2y2=False, GIoU=True)  # giou(prediction, target)
            giou = giou * tobjscores[i] # objectness applied to iou loss
            lbox += (1.0 - giou).sum() if red == 'sum' else (1.0 - giou).mean()  # giou loss

            # Obj
            tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * giou.detach().clamp(0).type(tobj.dtype)  # giou ratio

            # Class
            if model.nc > 1:  # cls loss (only if multiple classes)
                t = torch.full_like(ps[:, 5:], cn)  # targets
                t[range(nb), tcls[i]] = cp
                tmploss = BCEcls(ps[:, 5:], t).mean(dim=1)
                tmploss = tmploss * tobjscores[i] # objectness applied to classification loss
                lcls += tmploss.mean()  # BCE

            # Append targets to text file
            # with open('targets.txt', 'a') as file:
            #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

        lobj += BCEobj(pi[..., 4], tobj)  # obj loss

    lbox *= h['giou']
    lobj *= h['obj']
    lcls *= h['cls']
    if red == 'sum':
        bs = tobj.shape[0]  # batch size
        g = 3.0  # loss gain
        lobj *= g / bs
        if nt:
            lcls *= g / nt / model.nc
            lbox *= g / nt

    loss = lbox + lobj + lcls
    return loss, torch.cat((lbox, lobj, lcls, loss)).detach()