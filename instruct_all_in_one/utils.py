import numpy as np
import torch as th
import PIL
import cv2
import torchvision.transforms as T
from torch import nn
from PIL import Image, ImageOps
from torch.nn import init
import os, argparse, random
from torchvision import models, transforms

def dynamic_clip(x):
    if x > 10:
        x = 0.2 * x
    return x

class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x ):

        b,c,h,w = x.shape

        mean_rgb = th.mean(x,[2,3],keepdim=True)
        mr,mg, mb = th.split(mean_rgb, 1, dim=1)
        Drg = th.pow(mr-mg,2)
        Drb = th.pow(mr-mb,2)
        Dgb = th.pow(mb-mg,2)
        k = th.pow(th.pow(Drg,2) + th.pow(Drb,2) + th.pow(Dgb,2),0.5)

        return k.squeeze()

class L_exp(nn.Module):

    def __init__(self,patch_size,mean_val):
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val
    def forward(self, x ):

        b,c,h,w = x.shape
        x = th.mean(x,1,keepdim=True)
        mean = self.pool(x)

        d = th.mean(th.pow(mean- th.FloatTensor([self.mean_val] ).cuda(),2))
        return d

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input, target):
        return ((input - target) ** 2).mean()

class TVLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

    def forward(self, x):
        b_s, _, h_x, w_x = x.size()
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = th.pow((x[:,:,1:,:] - x[:,:,:h_x-1,:]),2).sum()
        w_tv = th.pow((x[:,:,:,1:] - x[:,:,:,:w_x-1]),2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / b_s

class VGGLoss(nn.Module):
    def __init__(self, layers=9):
        super().__init__()
        self.mse_loss = MSELoss()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.model = models.vgg16(pretrained=False)
        pre = th.load('/mnt/workspace/lhq/all-in-one/2024-ICML-TAO/vgg/vgg16-397923af.pth')
        self.model.load_state_dict(pre)
        self.model = self.model.features[:layers]
        #self.model = models.vgg16(
            #weights=models.VGG16_Weights.IMAGENET1K_V1).features[:layers]
        self.model.requires_grad_(False)
        self.model.eval()
    
    def forward(self, input, target):
        batch = th.cat([input, target], dim=0)
        feats = self.model(self.normalize(batch))
        input_feats, target_feats = feats.chunk(2, dim=0)
        return self.mse_loss(input_feats, target_feats)

class GANLoss(nn.Module):
    def __init__(self, lr = 4e-5):
        super().__init__()
        self.loss = nn.BCELoss()
        self.disc_net = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1), nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.LeakyReLU(),
            nn.Conv2d(256, 1, 1), nn.Sigmoid())
        self.optimizer = th.optim.Adam(self.disc_net.parameters(), lr)
    
    def init_disc_net(self):
        for m in self.disc_net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                if m.bias is not None: m.bias.data.fill_(0)
    
    def get_loss_value(self, input, is_real):
        target_value = 1.0 if is_real else 0.0
        target_label = th.ones_like(input) * target_value
        return self.loss(input, target_label)
    
    def forward(self, input, target):
        self.disc_net.train()
        self.optimizer.zero_grad()
        logits = self.disc_net(th.cat((input,target)).detach())
        fake_loss = self.get_loss_value(logits[:input.size(0)], False)
        real_loss = self.get_loss_value(logits[input.size(0):], True)
        (fake_loss + real_loss).backward()
        self.optimizer.step()
        self.disc_net.eval()
        return self.get_loss_value(self.disc_net(input), True)

class linear_degradation_net(nn.Module):
    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int = 3,
        embed_dim: int = 64,
        n_groups: int = 8,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(inp_channels, embed_dim, kernel_size, stride=1, padding=kernel_size//2),
            #nn.GroupNorm(n_groups, embed_dim), 
            nn.LeakyReLU(),
            nn.Conv2d(embed_dim, out_channels, kernel_size = 1),
            )
        self.net2 = nn.Sequential(
            nn.Conv2d(inp_channels, embed_dim, kernel_size, stride=1, padding=kernel_size//2),
            #nn.GroupNorm(n_groups, embed_dim), 
            nn.LeakyReLU(),
            nn.Conv2d(embed_dim, embed_dim * 2, kernel_size, stride=1, padding=kernel_size//2),
            #nn.GroupNorm(n_groups, embed_dim * 2), 
            nn.LeakyReLU(),
            nn.Conv2d(embed_dim * 2, embed_dim * 4, kernel_size, stride=1, padding=kernel_size//2),
            #nn.GroupNorm(n_groups, embed_dim * 2), 
            nn.LeakyReLU(),
            nn.Conv2d(embed_dim * 4, out_channels, kernel_size = 1),
            )
        self.skip_scale1= nn.Parameter(th.ones(1,inp_channels,1,1))
        #self.skip_scale2= nn.Parameter(th.ones(1,inp_channels,1,1))
    
    def forward(self, x):
        out1 = self.net1(x)
        out2 = self.net2(out1) + out1 * self.skip_scale1
        return out2

class GenerativeDegradation_res(nn.Module):
    def __init__(self, input_para, lr=4e-5, learn_rate = 5e-4):
        super().__init__()
        self.loss_dict = None
        self.mse_loss = MSELoss()
        self.perceptual_loss = VGGLoss()
        self.adversarial_loss = GANLoss(lr=lr)
        self.w_mse, self.w_pse, self.w_adv = input_para
        self.gene_net = linear_degradation_net()
        self.optimizer = th.optim.Adam(self.gene_net.parameters(), learn_rate, weight_decay=1e-5)
    
    def init_gene_net(self):
        for m in self.gene_net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                if m.bias is not None: m.bias.data.fill_(0)
        self.adversarial_loss.init_disc_net()
    
    def forward(self, input, target, t, TRAIN = True):
        self.loss_dict = {}
        if TRAIN:
            self.gene_net.train()
            self.optimizer.zero_grad()
            output = self.gene_net(input.detach())
            """ mse = self.mse_loss(output, input.detach()) * 0.02
            pse = self.perceptual_loss(output[:, :3], input[:, :3].detach()) * 0.3
            pse = th.clamp(pse, 0., 20.)
            adv = self.adversarial_loss(output, target.detach()) * 0.7 """
            # lowlight[0.02,0.3,0.7] dehazing[0.2,0.3,0.6]
            mse = self.mse_loss(output, input.detach()) * self.w_mse
            pse = self.perceptual_loss(output[:, :3], input[:, :3].detach()) * self.w_pse
            adv = self.adversarial_loss(output, target.detach()) * self.w_adv
            if t < 700: pse = dynamic_clip(pse)
                
            (mse + pse + adv).backward()
            self.optimizer.step()
            self.loss_dict = {"mse": mse, "pse": pse, "adv": adv}
        self.gene_net.eval()
        
        return self.gene_net(input)

class GenerativeDegradation_bn(nn.Module):
    def __init__(self, input_para):
        super().__init__()
        self.loss_dict = None
        self.mse_loss = MSELoss()
        self.perceptual_loss = VGGLoss()
        self.adversarial_loss = GANLoss()
        self.w_mse, self.w_pse, self.w_adv = input_para
        self.gene_net = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.LeakyReLU(),
            nn.Conv2d(256, 3, 1))
        self.optimizer = th.optim.Adam(self.gene_net.parameters(), 1e-3)
    
    def init_gene_net(self):
        for m in self.gene_net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                if m.bias is not None: m.bias.data.fill_(0)
        self.adversarial_loss.init_disc_net()
    
    def forward(self, input, target, t):
        self.gene_net.train()
        self.optimizer.zero_grad()
        output = self.gene_net(input.detach())
        """ mse = self.mse_loss(output, input.detach()) * 0.02
        pse = self.perceptual_loss(output[:, :3], input[:, :3].detach()) * 0.3
        pse = th.clamp(pse, 0., 20.)
        adv = self.adversarial_loss(output, target.detach()) * 0.7 """
        # lowlight[0.02,0.3,0.7] dehazing[0.2,0.3,0.6]
        mse = self.mse_loss(output, input.detach()) * self.w_mse
        pse = self.perceptual_loss(output[:, :3], input[:, :3].detach()) * self.w_pse
        adv = self.adversarial_loss(output, target.detach()) * self.w_adv
        if t < 700: pse = dynamic_clip(pse)
            
        (mse + pse + adv).backward()
        self.optimizer.step()
        self.gene_net.eval()
        self.loss_dict = {"mse": mse, "pse": pse, "adv": adv}
        return self.gene_net(input)

class GenerativeDegradation(nn.Module):
    def __init__(self, input_para, lr=1e-3):
        super().__init__()
        self.loss_dict = None
        self.mse_loss = MSELoss()
        self.perceptual_loss = VGGLoss()
        self.adversarial_loss = GANLoss(lr=lr)
        self.w_mse, self.w_pse, self.w_adv = input_para
        self.gene_net = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 1, 1), nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 1, 1), nn.LeakyReLU(),
            nn.Conv2d(256, 3, 1))
        self.optimizer = th.optim.Adam(self.gene_net.parameters(), 5e-4)
    
    def init_gene_net(self):
        for m in self.gene_net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                if m.bias is not None: m.bias.data.fill_(0)
        self.adversarial_loss.init_disc_net()
    
    def forward(self, input, target, t):
        self.gene_net.train()
        self.optimizer.zero_grad()
        output = self.gene_net(input.detach())
        """ mse = self.mse_loss(output, input.detach()) * 0.02
        pse = self.perceptual_loss(output[:, :3], input[:, :3].detach()) * 0.3
        pse = th.clamp(pse, 0., 20.)
        adv = self.adversarial_loss(output, target.detach()) * 0.7 """
        # lowlight[0.02,0.3,0.7] dehazing[0.2,0.3,0.6]
        mse = self.mse_loss(output, input.detach()) * self.w_mse
        pse = self.perceptual_loss(output[:, :3], input[:, :3].detach()) * self.w_pse
        adv = self.adversarial_loss(output, target.detach()) * self.w_adv
        if t < 700: pse = dynamic_clip(pse)
            
        (mse + pse + adv).backward()
        self.optimizer.step()
        self.gene_net.eval()
        self.loss_dict = {"mse": mse, "pse": pse, "adv": adv}
        return self.gene_net(input)

def load_img(path, resize = 512):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((resize, resize), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = th.from_numpy(image)
    #image = torch.pow(image, 0.4)
    return 2.*image - 1.

def load_img2img(path, resize = 512):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    #image = image.crop((0, 0, 512, 400))
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = th.from_numpy(image)
    pad_left, pad_bottom = 0, 0
    if w < resize:
        pad_left = resize - w
        image = th.nn.functional.pad(
            image, pad=(resize - w, 0, 0, 0), mode="reflect"
        )
    if h < resize:
        pad_bottom = resize - h
        image = th.nn.functional.pad(
            image, pad=(0, 0, 0, resize - h), mode="reflect"
        )
    return 2.*image - 1., pad_left, pad_bottom

def load_fre_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((512,512), resample=PIL.Image.LANCZOS)
    image = np.array(image)
    image = image.astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = th.from_numpy(image)
    return 2.*image - 1.

def median_filter(img, kernel_size=11):
    # 对图像应用中值滤波
    img_denoised = cv2.medianBlur(img, kernel_size)
    return img_denoised