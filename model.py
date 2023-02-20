
import torch as T

VGG = T.nn.Sequential(
    T.nn.Conv2d(3, 3, 1),
    T.nn.ReflectionPad2d(1), T.nn.Conv2d(3, 64, 3), T.nn.ReLU(), 
    T.nn.ReflectionPad2d(1), T.nn.Conv2d(64, 64, 3), T.nn.ReLU(), 
    T.nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True), 
    T.nn.ReflectionPad2d(1), T.nn.Conv2d(64, 128, 3), T.nn.ReLU(), 
    T.nn.ReflectionPad2d(1), T.nn.Conv2d(128, 128, 3), T.nn.ReLU(), 
    T.nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    T.nn.ReflectionPad2d(1), T.nn.Conv2d(128, 256, 3), T.nn.ReLU(), 
    T.nn.ReflectionPad2d(1), T.nn.Conv2d(256, 256, 3), T.nn.ReLU(), 
    T.nn.ReflectionPad2d(1), T.nn.Conv2d(256, 256, 3), T.nn.ReLU(), 
    T.nn.ReflectionPad2d(1), T.nn.Conv2d(256, 256, 3), T.nn.ReLU(), 
    T.nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    T.nn.ReflectionPad2d(1), T.nn.Conv2d(256, 512, 3), T.nn.ReLU(), 
    T.nn.ReflectionPad2d(1), T.nn.Conv2d(512, 512, 3), T.nn.ReLU(), 
    T.nn.ReflectionPad2d(1), T.nn.Conv2d(512, 512, 3), T.nn.ReLU(), 
    T.nn.ReflectionPad2d(1), T.nn.Conv2d(512, 512, 3), T.nn.ReLU(), 
    T.nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    T.nn.ReflectionPad2d(1), T.nn.Conv2d(512, 512, 3), T.nn.ReLU(), 
    T.nn.ReflectionPad2d(1), T.nn.Conv2d(512, 512, 3), T.nn.ReLU(), 
    T.nn.ReflectionPad2d(1), T.nn.Conv2d(512, 512, 3), T.nn.ReLU(), 
    T.nn.ReflectionPad2d(1), T.nn.Conv2d(512, 512, 3), T.nn.ReLU()
)
for p in VGG.parameters(): p.requires_grad = False

DECODER = T.nn.Sequential(
    T.nn.ReflectionPad2d(1), T.nn.Conv2d(512, 256, 3), T.nn.ReLU(),
    T.nn.Upsample(scale_factor=2, mode='nearest'),
    T.nn.ReflectionPad2d(1), T.nn.Conv2d(256, 256, 3), T.nn.ReLU(),
    T.nn.ReflectionPad2d(1), T.nn.Conv2d(256, 256, 3), T.nn.ReLU(),
    T.nn.ReflectionPad2d(1), T.nn.Conv2d(256, 256, 3), T.nn.ReLU(),
    T.nn.ReflectionPad2d(1), T.nn.Conv2d(256, 128, 3), T.nn.ReLU(),
    T.nn.Upsample(scale_factor=2, mode='nearest'),
    T.nn.ReflectionPad2d(1), T.nn.Conv2d(128, 128, 3), T.nn.ReLU(),
    T.nn.ReflectionPad2d(1), T.nn.Conv2d(128, 64, 3), T.nn.ReLU(),
    T.nn.Upsample(scale_factor=2, mode='nearest'),
    T.nn.ReflectionPad2d(1), T.nn.Conv2d(64, 64, 3), T.nn.ReLU(), 
    T.nn.ReflectionPad2d(1),
    T.nn.Conv2d(64, 3, 3)
)

class SA(T.nn.Module):
    def norm(self, feat, eps=1e-5):
        B, C = feat.shape[:2]
        var = feat.view([B, C, -1]).var(dim=2) + eps
        std = var.sqrt().view([B, C, 1, 1])
        mean = feat.view([B, C, -1]).mean(dim=2).view([B, C, 1, 1])
        norm = (feat-mean.expand(feat.shape)) / std.expand(feat.shape)
        return norm
    
    def __init__(self, c):
        super().__init__()
        
        self.f, self.g, self.h = T.nn.Conv2d(c, c, 1), T.nn.Conv2d(c, c, 1), T.nn.Conv2d(c, c, 1)
        self.cnn = T.nn.Conv2d(c, c, 1)
    
    def forward(self, con, sty):
        f, g, h = self.f(self.norm(con)), self.g(self.norm(sty)), self.h(sty)
        
        [B, _, H_f, W_f], [_, _, H_g, W_g] = f.shape, g.shape
        a = T.bmm(f.view([B, -1, W_f*H_f]).permute(0, 2, 1), g.view([B, -1, W_g*H_g]))
        a = T.nn.functional.softmax(a, dim=-1)
        o = T.bmm(h.view([B, -1, W_g*H_g]), a.permute(0, 2, 1))
        
        _, C_c, H_c, W_c = con.shape
        o = con + self.cnn(o.view([B, C_c, H_c, W_c]))
        
        return o

class CLVA(T.nn.Module):
    def __init__(self, c):
        super().__init__()
        
        self.enc_c4, self.enc_c5 = T.nn.Sequential(*list(VGG.children())[:31]), T.nn.Sequential(*list(VGG.children())[31:44])
        self.enc_s4, self.enc_s5 = [T.nn.Sequential(*[T.nn.Linear(512, 4096), T.nn.ReLU(), 
                                                      T.nn.Linear(4096, 512*8*8), T.nn.ReLU()]), 
                                    T.nn.Sequential(*[T.nn.Conv2d(512, 512, 3, padding=1), T.nn.ReLU(), 
                                                      T.nn.MaxPool2d(2), 
                                                      T.nn.Conv2d(512, 512, 3, padding=1), T.nn.ReLU()])]
        
        self.sa4, self.sa5 = SA(c), SA(c)
        self.pad, self.cnn = T.nn.ReflectionPad2d(1), T.nn.Conv2d(c, c, 3)
        
        self.dec = DECODER
    
    def fusion(self, c4, s4, c5, s5):
        sa4, sa5 = self.sa4(c4, s4), self.sa5(c5, s5)
        sa = self.pad(sa4 + T.nn.functional.interpolate(sa5, size=[c4.shape[2], c4.shape[3]], mode='nearest')) # bicubic
        sa = self.cnn(sa)
        
        out = self.dec(sa)
        
        return out
    
    def forward(self, con, ins): # forward_cx
        B = con.shape[0]
        
        c4, s4 = self.enc_c4(con), self.enc_s4(ins).view([B, -1, 8, 8])
        c5, s5 = self.enc_c5(c4), self.enc_s5(s4)
        
        F = min(c4.shape[-2]//s4.shape[-2], c5.shape[-1]//s5.shape[-1])
        s4, s5 = [T.nn.functional.interpolate(s4, scale_factor=F, mode='bicubic', align_corners=True), 
                  T.nn.functional.interpolate(s5, scale_factor=F, mode='bicubic', align_corners=True)]
        
        out = self.fusion(c4, s4, c5, s5)
        
        return out
    
    def forward_cs(self, con, sty):
        B = con.shape[0]
        
        c4, s4 = self.enc_c4(con), self.enc_c4(sty)
        c5, s5 = self.enc_c5(c4), self.enc_c5(s4)
        
        out = self.fusion(c4, s4, c5, s5)
        
        return out

class Discriminator(T.nn.Module): 
    def __init__(self):
        super().__init__()
        
        self.enc = T.nn.Sequential(*list(VGG.children())[:31])
        self.cnn = T.nn.Sequential(*[T.nn.ReflectionPad2d(1), T.nn.Conv2d(512, 512, 3), T.nn.ReLU(), 
                                     T.nn.ReflectionPad2d(1), T.nn.Conv2d(512, 512, 3), T.nn.ReLU(), 
                                     T.nn.AdaptiveAvgPool2d(1)])
        self.fc = T.nn.Sequential(*[T.nn.Linear(1024, 1024), T.nn.ReLU(), 
                                    T.nn.Linear(1024, 1), T.nn.Sigmoid()])
        
    def forward(self, patch, ins):
        f = self.enc(patch)
        f = self.cnn(f).squeeze()
        out = self.fc(T.cat([f, ins], dim=1))
        
        return out
    
