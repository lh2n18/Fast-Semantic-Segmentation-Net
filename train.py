import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import os
import sys
import argparse
import time
import PIL.Image as Image
from models_seg import SegNet, weights_init_normal
from dataset import SegDataset

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("--cuda", type=bool, default=True, help="number of gpu")
parser.add_argument("--gpu_num", type=int, default=1, help="number of gpu")
parser.add_argument("--worker_num", type=int, default=4, help="number of input workers")
parser.add_argument("--batch_size", type=int, default=4, help="batch size of input")
parser.add_argument("--lr", type=float, default=0.01, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--begin_epoch", type=int, default=0, help="begin_epoch")
parser.add_argument("--end_epoch", type=int, default=21, help="end_epoch")
parser.add_argument("--need_test", type=bool, default=True, help="need to test")
parser.add_argument("--test_interval", type=int, default=2, help="interval of test")
parser.add_argument("--need_save", type=bool, default=True, help="need to save")
parser.add_argument("--save_interval", type=int, default=1, help="interval of save weights")
parser.add_argument("--img_height", type=int, default=512, help="size of image height") 
parser.add_argument("--img_width", type=int, default=512, help="size of image width")
opt = parser.parse_args()
print(opt)

# Root Dir
dataSetRoot = "./data" 

# Build nets
seg_net = SegNet(init_weights=True)

# Loss functions
criterion_seg  = torch.nn.BCEWithLogitsLoss() # if the background pixels are far more，y=0 will dominate the Loss function，leading to a deviated model.
'''
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__() 
    def forward(self, input, target):
        N = target.size(0)
        smooth = 1
        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)
        intersection = input_flat * target_flat
        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N 
        return loss
'''

# Optimizers
optimizer_seg = torch.optim.Adam(seg_net.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
# optimizer_seg = torch.optim.SGD(seg_net.parameters(),lr=opt.lr,momentum=0.9)# momentum accelerate fitting;weight_decay=L2 penalty,reduce overfitting.

# Schedulers
scheduler_seg = ReduceLROnPlateau(optimizer_seg, mode='min', factor=0.1, patience=0, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-08, eps=1e-08)
# scheduler_seg = StepLR(optimizer_seg, step_size=3, gamma=0.1)

# GPU Settings
if opt.cuda:
    seg_net = seg_net.cuda()
    criterion_seg.cuda()
if opt.gpu_num > 1:
    seg_net = torch.nn.DataParallel(seg_net, device_ids=list(range(opt.gpu_num)))
    
# Weights Settings
if opt.begin_epoch != 0:
    seg_net.load_state_dict(torch.load("./saved_models_10/SegNet_Adam_%d.pth" % (opt.begin_epoch))) # Load pretrained models
else:
    seg_net.apply(weights_init_normal) # Initialize weights

# DataLoader
transforms_ = transforms.Compose([
    transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
transforms_mask = transforms.Compose([
    transforms.Resize((opt.img_height//1, opt.img_width//1)),
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
trainloader = DataLoader(
    SegDataset(dataSetRoot, transforms_=transforms_, transforms_mask= transforms_mask, subFold="train", isTrain=True),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.worker_num,
)
testloader = DataLoader(
    SegDataset(dataSetRoot, transforms_=transforms_, transforms_mask= transforms_mask, subFold="test", isTrain=False),
    batch_size=1,
    shuffle=False,
    num_workers=opt.worker_num,
)

def Iou(seg,target,classNum):

    segTmp = torch.zeros([seg.shape[0],classNum,seg.shape[2],seg.shape[3]]) # [b,c,h,w] matrics
    targetTmp = torch.zeros([target.shape[0],classNum,target.shape[2],target.shape[3]])#[b,c,h,w] matrics
    seg_cpu = seg.data.cpu()
    seg_cpu = np.where(seg_cpu.data > 0, 1, 0)
    seg_tensor = torch.from_numpy(seg_cpu)
    target_cpu = target.data.cpu()
    target_cpu = np.where(target_cpu.data.cpu() > 0.5, 1, 0)
    target_tensor = torch.from_numpy(target_cpu)
    segOht = segTmp.scatter_(1, torch.LongTensor(seg_tensor), 1) # dim index value
    targetOht = targetTmp.scatter_(1, torch.LongTensor(target_tensor), 1) # dim index value
    batchMious = [] # batch miou for n images
    mul = segOht * targetOht # intersection = numbers of 1
    for i in range(seg.shape[0]): # per image
        ious = []
        for j in range(classNum): # n classes including background
            intersection = torch.sum(mul[i][j])
            union = torch.sum(segOht[i][j]) + torch.sum(targetOht[i][j]) - intersection + 1e-6
            iou = intersection / union
            ious.append(iou)
        miou = np.mean(ious) # miou
        batchMious.append(miou)
    return np.mean(batchMious)

# Training and inference
for epoch in range(opt.begin_epoch, opt.end_epoch):
    print("\nEpoch %d training......" % epoch)
    print("\nLearning rate ：%f" % optimizer_seg.state_dict()['param_groups'][0]['lr'])    
    iterImg = trainloader.__iter__()
    lenNum = len(trainloader) 
    
    # train 
    seg_net.train() 
    for i in range(0, lenNum):
        batchData = iterImg.__next__()
        if opt.cuda:
            img = batchData["img"].cuda()
            mask = batchData["mask"].cuda()
        else:
            img = batchData["img"]
            mask = batchData["mask"] 
        optimizer_seg.zero_grad()
        out = seg_net(img)
        seg = out["seg"]
        miou_seg = Iou(seg,mask,2) # MIOU for training batchs
        loss_seg = criterion_seg(seg, mask)
        loss_seg.backward()
        optimizer_seg.step() 
        sys.stdout.write(
            "\r [Epoch %d/%d]  [Batch %d/%d] [loss %f] [MIOU %f]"
             %(
                epoch,
                opt.end_epoch,
                i,
                lenNum,
                loss_seg.item(),
                miou_seg
             )
        )

    # Record Training Loss
    with open("loss-Adam.txt","a+") as file:
        file.write(str(optimizer_seg.state_dict()['param_groups'][0]['lr']) + "  " + str(loss_seg.item()) + "\n")
        
    # test 
    if opt.need_test and epoch % opt.test_interval == 0 and epoch >= opt.test_interval:
        seg_net.eval()
        all_time = 0
        miou = []
        for i, testBatch in enumerate(testloader):
            imgTest = testBatch["img"].cuda()
            maskTest = testBatch["mask"].cuda()
            t1 = time.time()
            outTest = seg_net(imgTest)
            segTest = outTest["seg"]
            t2 = time.time()
            miouTest = Iou(segTest,maskTest,2) # MIOU for val batchs
            miou.append(miouTest)
            save_path_str = "./Results-10/Epoch_Adam_%d"%epoch
            if os.path.exists(save_path_str) == False:
                os.makedirs(save_path_str, exist_ok=True)
            save_image(imgTest.data, "%s/Img_%d.png"% (save_path_str, i))
            save_image(segTest.data, "%s/Img_%d_seg.png"% (save_path_str, i))
            all_time = (t2-t1) + all_time
            count_time = i + 1
        avg_time = all_time/count_time
        print("\n****** %fs per image ******" % avg_time)
        miou_batch = np.mean(miou)
        print("\n****** Val MIOU : %f ******" % miou_batch)
        with open("MIOU-Adam.txt","a+") as file:
            file.write(str(miou_batch) + "\n")
        seg_net.train()
        
    scheduler_seg.step(loss_seg) # update lr each epoch according to strategy
    
    # save model parameters 
    if opt.need_save and epoch % opt.save_interval == 0 and epoch >= opt.save_interval:
        save_path_str = "./saved_models_10"
        if os.path.exists(save_path_str) == False:
            os.makedirs(save_path_str, exist_ok=True)
        torch.save(seg_net.state_dict(), "%s/SegNet_Adam_%d.pth" % (save_path_str, epoch))
        print(" Weights saved ! epoch = %d"%epoch)
        pass
