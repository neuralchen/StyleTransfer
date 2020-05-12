import torch

from torch import nn
from torch.nn import init
from torch.nn import functional as F

from torch.nn import utils

class Discriminator(nn.Module):
    def __init__(self, chn=32, k_size=3,n_class=3):
        super().__init__()
        # padding_size = int((k_size -1)/2)
        slop         = 0.2
        enable_bias  = True



        # stage 1
        self.block1 = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(in_channels = 3, out_channels = chn,
                                kernel_size= k_size, stride = 2, bias= enable_bias)), # 1/2
            nn.LeakyReLU(slop),
        )
        self.aux_classfier1 = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(in_channels = chn, out_channels = chn,
                                kernel_size= 5, bias=enable_bias)),
            nn.LeakyReLU(slop),
            nn.AdaptiveAvgPool2d(1),
        )
        self.linear1= utils.spectral_norm(nn.Linear(chn, n_class+1))

        # stage 2
        self.block2 = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(in_channels = chn, out_channels = chn * 2,
                                kernel_size= k_size, stride = 2, bias= enable_bias)), # 1/4
            nn.LeakyReLU(slop),
            utils.spectral_norm(nn.Conv2d(in_channels = chn * 2, out_channels = chn * 4,
                                kernel_size= k_size, stride = 2, bias= enable_bias)),# 1/8
            nn.LeakyReLU(slop),
        )
        self.aux_classfier2 = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(in_channels = chn * 4, out_channels = chn,
                                kernel_size= 5, bias= enable_bias)),
            nn.LeakyReLU(slop),
            nn.AdaptiveAvgPool2d(1),
        )
        self.linear2= utils.spectral_norm(nn.Linear(chn, n_class+1))

        # stage 3
        self.block3 = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(in_channels = chn * 4, out_channels = chn * 8,
                                kernel_size= k_size, stride = 2, bias= enable_bias)),# 1/16
            nn.LeakyReLU(slop),
            utils.spectral_norm(nn.Conv2d(in_channels = chn * 8 , out_channels = chn * 16,
                                kernel_size= k_size, stride = 2, bias= enable_bias)),# 1/32
            nn.LeakyReLU(slop)
        )
        self.aux_classfier3 = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(in_channels = chn * 16, out_channels = chn,
                                kernel_size= 5, bias= enable_bias)),
            nn.LeakyReLU(slop),
            nn.AdaptiveAvgPool2d(1),
        )
        self.linear3= utils.spectral_norm(nn.Linear(chn, n_class+1))
        self.__weights_init__()

    def __weights_init__(self):
        print("Init weights")
        for m in self.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                try:
                    nn.init.zeros_(m.bias)
                except:
                    print("No bias found!")

    def forward(self, input,feat=False):
        
        h       = self.block1(input)
        # if feat:
        #     out1    = torch.sum(h, [2, 3])
        # else:
        #     prep1   = self.aux_classfier1(h)
        #     prep1   = prep1.view(prep1.size()[0], -1)
        #     prep1   = self.linear1(prep1)

        # h       = self.block2(h)
        # if feat:
        #     out2    = torch.sum(h, [2, 3])
        # else:
        #     prep2   = self.aux_classfier2(h)
        #     prep2   = prep2.view(prep2.size()[0], -1)
        #     prep2   = self.linear2(prep2)

        # h       = self.block3(h)
        # if feat:
        #     out3    = torch.sum(h, [2, 3])
        # else:
        #     prep3   = self.aux_classfier3(h)
        #     prep3   = prep3.view(prep3.size()[0], -1)
        #     prep3   = self.linear3(prep3)

        # if feat:
        #     out_list=[out1,out2,out3]
        #     return out_list
        # else:

        #     out_prep = [prep1,prep2,prep3]
        #     return out_prep


        if not feat:
            prep1   = self.aux_classfier1(h)
            prep1   = prep1.view(prep1.size()[0], -1)
            prep1   = self.linear1(prep1)

        h       = self.block2(h)
        if not feat:
            prep2   = self.aux_classfier2(h)
            prep2   = prep2.view(prep2.size()[0], -1)
            prep2   = self.linear2(prep2)

        h       = self.block3(h)
        if not feat:
            prep3   = self.aux_classfier3(h)
            prep3   = prep3.view(prep3.size()[0], -1)
            prep3   = self.linear3(prep3)
            out_prep = [prep1,prep2,prep3]
            return out_prep
        else:
            out    = torch.sum(h, [2, 3])
            return out


        # h       = self.block1(input)
        # prep1   = self.aux_classfier1(h)
        # prep1   = prep1.view(prep1.size()[0], -1)
        # prep1   = self.linear1(prep1)

        # h       = self.block2(h)
        # prep2   = self.aux_classfier2(h)
        # prep2   = prep2.view(prep2.size()[0], -1)
        # prep2   = self.linear2(prep2)

        # h       = self.block3(h)
        # prep3   = self.aux_classfier3(h)
        # prep3   = prep3.view(prep3.size()[0], -1)
        # prep3   = self.linear3(prep3)

        # out_prep = [prep1,prep2,prep3]
        # return out_prep
    
    def get_outputs_len(self):
        num = 0
        for m in self.modules():
            if isinstance(m,nn.Linear):
                num+=1
        return num

if __name__ == "__main__":
    wocao = Discriminator().cuda()
    from torchsummary import summary
    summary(wocao, input_size=(3, 512, 512))