import torch.nn as nn
import torch
import torch.nn.functional as F


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02) # N(mean, std)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0) # Constant
    elif classname.find("Linear") != -1:
        torch.nn.init.constant_(m.weight.data, 0.0)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)


class SegNet(nn.Module):
    def __init__(self, in_channels=3, init_weights=True):
        super(SegNet, self).__init__()

        # Conv Block
        self.Conv = nn.Sequential(  # stride=1,padding=0,dilation=1,groups=1,bias=True by default.
                            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
                            nn.BatchNorm2d(32),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(2)  
                        )
        # Se Module
        self.Avg_Pool = nn.AdaptiveAvgPool2d(1)
        self.Se = nn.Sequential(
                            nn.Linear(64, 64 // 2, bias=False),
                            nn.ReLU(inplace=True),
                            nn.Linear(64 // 2, 64, bias=False),
                            nn.Sigmoid(),
                        )
        # Depthwise + Pointwise Blocks
        self.DS_Conv1 = nn.Sequential(                                     
                            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64, bias=False),
                            nn.BatchNorm2d(64),
                            nn.Conv2d(64, 128, kernel_size=1, bias=False),
                            nn.BatchNorm2d(128),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128, bias=False),
                            nn.BatchNorm2d(128),
                            nn.Conv2d(128, 256, kernel_size=1, bias=False),
                            nn.BatchNorm2d(256),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(2)
                        )    
        self.DS_Conv2 = nn.Sequential(                                     
                            nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=256, bias=False),
                            nn.BatchNorm2d(256),
                            nn.Conv2d(256, 512, kernel_size=1, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(512, 512, kernel_size=3, padding=1, groups=512, bias=False),
                            nn.BatchNorm2d(512),
                            nn.Conv2d(512, 1024, kernel_size=1, bias=False),
                            nn.BatchNorm2d(1024),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(2)
                        )  
        # BottleNecks
        self.Neck1 = nn.Sequential(
                            nn.Conv2d(1024, 256, kernel_size=1, bias=False),
                            nn.BatchNorm2d(256),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(512, 128, kernel_size=1, bias=False),
                            nn.BatchNorm2d(128),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
                            nn.BatchNorm2d(256),
                            nn.ReLU(inplace=True),
                        )
        self.Neck2 = nn.Sequential(
                            nn.Conv2d(256, 64, kernel_size=1, bias=False),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
                            nn.BatchNorm2d(128),
                            nn.ReLU(inplace=True),
                        )
        self.Neck3 = nn.Sequential(
                            nn.Conv2d(64, 32, kernel_size=1, bias=False),
                            nn.BatchNorm2d(32),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                        )
        self.Neck4 = nn.Sequential(
                            nn.Conv2d(64, 16, kernel_size=1, bias=False),
                            nn.BatchNorm2d(16),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
                            nn.BatchNorm2d(32),
                            nn.ReLU(inplace=True),
                        )
        # Upsample(Deconv)
        self.Up1 = nn.Sequential(   # stride=1,padding=0,output_padding=0,groups=1,bias=True,dilation=1,padding_mode='zeros' by default.
                            nn.ConvTranspose2d(in_channels=64, out_channels= 64, kernel_size=2, stride=2, bias= False),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                        )
        self.Up2 = nn.Sequential(
                            nn.ConvTranspose2d(in_channels=128, out_channels= 64, kernel_size=2, stride=2, bias= False),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                        )
        self.Up3 = nn.Sequential(
                            nn.ConvTranspose2d(in_channels=256, out_channels= 256, kernel_size=2, stride=2, bias= False),
                            nn.BatchNorm2d(256),
                            nn.ReLU(inplace=True),
                        )
        # Outputs
        self.Out_Conv = nn.Sequential(
                            nn.Conv2d(32, 1, 1),                           
                        )
        # Weights Initialization
        if init_weights == True:
            pass

    def forward(self, x):
        # Conv Block
        x1 = self.Conv(x) # 64c 1/2-size
        # Squeeze
        w = self.Avg_Pool(x1).view(x1.shape[0], x1.shape[1])
        w = self.Se(w).view(x1.shape[0], x1.shape[1], 1, 1)
        # Excitation
        x1 = x1 * w
        # Depth-Separable Conv Blocks
        x2 = self.DS_Conv1(x1) # 256c 1/4-size
        x3 = self.DS_Conv2(x2) # 1024c 1/8-size
        # BottleNeck & Upsample & Fusion
        x4 = self.Neck1(x3) # 1024c 1/8-size -> 256c 1/8-size
        x5 = self.Up3(x4) # 256c 1/8-size -> 256c 1/4-size
        x6 = self.Neck2(x2 + x5) # 256c 1/4-size -> 128c 1/4-size
        x7 = self.Up2(x6) # 128c 1/4-size -> 64c 1/2-size
        x8 = self.Neck3(x1 + x7) # 64c 1/2-size -> 64c 1/2-size
        x9 = self.Up1(x8) # 64c 1/2-size -> 64c 1-size
        x10 = self.Neck4(x9) # 64c 1-size -> 32c 1-size
        # classifier
        out = self.Out_Conv(x10)
        
        return {"seg":out}


if  __name__=='__main__':
    
    seg_net = SegNet()
    img = torch.randn(4, 3, 768, 768)

    seg_net.eval()
    seg_net = seg_net.cuda()
    img = img.cuda()

    out = seg_net(img)
    s = out["seg"]
    print(s)

    pass



