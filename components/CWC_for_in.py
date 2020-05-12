import torch
from torch import nn
import numpy as np

class WC(nn.Module):
    def __init__(self, in_channels, number_of_classes=3,
                  momentum=0.99,
                  epsilon=1e-3,
                  decomposition='cholesky',
                  is_training=True):
        super().__init__()
        self.whiting=Whiting(in_channels,momentum,epsilon,decomposition,is_training)
        self.coloring= Coloring(number_of_classes,in_channels)
    def forward(self,x,class_id):
        x=self.whiting(x)
        out=self.coloring(x,class_id)
        return out

class Whiting(nn.Module):
    def __init__(self,
                  in_channel,
                  momentum=0.99,
                  epsilon=1e-3,
                  decomposition='cholesky',
                  is_training=True,
                  num_pergroup=32):
        super().__init__()
        self.supports_masking = True
        self.momentum = momentum
        self.epsilon = epsilon
        self.in_channel=in_channel
        self.num_pergroup = num_pergroup
        assert in_channel % num_pergroup == 0
        self.num_groups = in_channel // num_pergroup

        self.is_training=is_training
        self.decomposition =decomposition
        self.minusmom_1=torch.tensor((1.0 - self.momentum), requires_grad=False).cuda()
        self.mom = torch.tensor(self.momentum, requires_grad=False).cuda()

        self.minuseps=torch.tensor((1 - self.epsilon), requires_grad=False).cuda()
        self.eps = torch.tensor(self.epsilon, requires_grad=False).cuda()
        #self.eye_icbs = torch.eye(self.in_channel*self.batch_size, requires_grad=False).cuda()



        #self.eye_icbs = torch.ones((self.batch_size*self.num_groups, self.num_pergroup, self.num_pergroup), requires_grad=False).cuda()

    def forward(self,x):
        N, C, H, W = x.size()
        c, g = self.num_pergroup, self.num_groups
        # x_flat=torch.transpose(x,1,0)
        # x_flat=x_flat.view(c,-1)
        x_flat = x.view(N * g, c, -1)

        eye = x_flat.data.new().resize_(c, c)
        self.eye_icbs = torch.nn.init.eye_(eye).view(1, c, c).expand(N * g, c, c)
        m      = x_flat.mean(-1, keepdim=True)
        if self.is_training:
            f = x_flat - m
            ff_apr = torch.matmul(f, f.permute(0,2,1)).div(H * W)
            #print(self.eye_icbs.size())
            #print(ff_apr.size())

            ff_apr_shrinked = self.minuseps * ff_apr + self.eye_icbs * self.eps
            if self.decomposition == 'cholesky':
                #print(ff_apr_shrinked[0].size())
                inv_sqrt=torch.cholesky(ff_apr_shrinked).inverse()
                #inv_sqrt=torch.solve(self.eye_icbs,sqrt)[0]
            elif self.decomposition == 'zca':
                U, S, _ = torch.svd(ff_apr_shrinked.cpu())
                U=U.cuda()
                S=S.cuda()
                D = torch.diag_embed(torch.pow(S, -0.5))
                print(torch.matmul(U,D).size())
                print(U.size())
                inv_sqrt = torch.matmul(torch.matmul(U,D),U.permute(0,2,1))

        # else:
        #     m=self.moving_mean
        #     f = x_flat - m
        #     ff_mov = torch.tensor(1 - self.epsilon).cuda() * self.moving_cov + torch.eye(self.in_channel) * self.epsilon
        #     if self.decomposition == 'cholesky':
        #         sqrt=torch.cholesky(ff_mov)
        #         inv_sqrt=torch.solve(torch.eye(self.in_channel*self.batch_size).cuda(),sqrt)
        

        # ff_apr = torch.matmul(f, f.permute(1,0)) / (bs*w*h.type(torch.FloatTensor)-1.)
        # ff_apr_shrinked = (1 - self.epsilon) * ff_apr + torch.eye(self.in_channel) * self.epsilon
        # if self.decomposition == 'cholesky':
        #     sqrt=torch.cholesky(ff_apr_shrinked)
        #     inv_sqrt=torch.solve(torch.eye(self.in_channel),sqrt)


        f_hat = torch.matmul(inv_sqrt, f)
        decorelated = f_hat.view(N,C,H,W)

        return decorelated
    
class Coloring(nn.Module):
    def __init__(self,
             number_of_classes,
             in_channels):
        super().__init__()
        self.in_channels= in_channels
        self.embedkernel = nn.Embedding(number_of_classes, in_channels*in_channels)
        self.embedbias = nn.Embedding(number_of_classes, in_channels)
        self.conv=nn.Conv2d(in_channels,in_channels,1,1)
        nn.init.xavier_uniform_(self.embedkernel.weight)
        nn.init.zeros_(self.embedbias.weight)

    def forward(self,x,class_id):
        uncond_output=self.conv(x)
        bs,c,w,h=x.size()
        x=x.view(bs,c,w*h)

        kernel   = self.embedkernel(class_id)
        bias     = self.embedbias(class_id)
        bias=torch.unsqueeze(bias,dim=-1)
        bias=torch.unsqueeze(bias,dim=-1)
        kernel=kernel.view(bs,self.in_channels,self.in_channels)

        cond_output=torch.matmul(x.permute(0,2,1),kernel)
        cond_output=cond_output.permute(0,2,1).view(bs,c,w,h)
        cond_output=cond_output+bias
        

        return cond_output+uncond_output




        



