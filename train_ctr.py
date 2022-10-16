
import io, base64, pickle

from tqdm import tqdm

import numpy as np
import torch as T
import torchvision as TV

from PIL import Image

import clip
from model import CLVA, Discriminator

def b2f(b): return Image.open(io.BytesIO(base64.b64decode(b))).convert('RGB')

class DS(T.utils.data.Dataset):
    def __init__(self, path):
        super().__init__()
        
        self.pkl = pickle.load(open(path, 'rb'))
        self.N_CONTENT, self.N_STYLE = 14425, 5369
        
    def __len__(self):
        return self.N_CONTENT*self.N_STYLE
    
    def __getitem__(self, idx):
        nc_1, ns_1 = idx//self.N_STYLE, idx%self.N_STYLE
        nc_2, ns_2 = np.random.randint(self.N_CONTENT), np.random.randint(self.N_STYLE)

        con_1, con_2 = TV.transforms.ToTensor()(b2f(self.pkl['content'][nc_1])), TV.transforms.ToTensor()(b2f(self.pkl['content'][nc_2]))
        sty_1, sty_2 = TV.transforms.ToTensor()(b2f(self.pkl['style'][ns_1]['image'])), TV.transforms.ToTensor()(b2f(self.pkl['style'][ns_2]['image']))
        ins_1, ins_2 = np.random.choice(self.pkl['style'][ns_1]['instruction']), np.random.choice(self.pkl['style'][ns_2]['instruction'])
        
        return con_1, sty_1, ins_1, con_2, sty_2, ins_2

def loss_percept(f1, f2):
    [B, C, H1, W1], [_, _, H2, W2] = f1.shape, f2.shape
    f1, f2 = f1.view([B*C, H1*W1]), f2.view([B*C, H2*W2])
    f1, f2 = T.mm(f1, f1.t())/(H1*W1), T.mm(f2, f2.t())/(H2*W2)
    ret = T.nn.functional.mse_loss(f1, f2)
    return ret    
    
if __name__=='__main__':
    dl = T.utils.data.DataLoader(DS('./_data/dtd.pkl'), batch_size=12, num_workers=6, shuffle=True, pin_memory=True)    
    
    CLIP, _ = clip.load('ViT-B/32', device='cuda')
    G = CLVA(512).cuda().train()
    G.load_state_dict(T.load('./_ckpt/lva.pt', map_location='cpu'))
    
    loss_mse = T.nn.MSELoss().cuda()
    optzr = T.optim.AdamW(G.parameters(), lr=0.0003, betas=(0.9, 0.98))
    print('G:', [(n, p.requires_grad) for n, p in G.named_parameters()])
    
    step, TQ = 0, tqdm(dl, ascii=True)
    while True:
        bh_cc, bh_cs, = [], []
        for con_1, sty_1, ins_1, con_2, sty_2, ins_2 in TQ:
            con_1, sty_1, con_2, sty_2 = con_1.cuda(), sty_1.cuda(), con_2.cuda(), sty_2.cuda()
            ins_1, ins_2 = [T.from_numpy(CLIP.encode_text(clip.tokenize(ins_1, truncate=True).cuda()).float().data.cpu().numpy()).cuda(), 
                            T.from_numpy(CLIP.encode_text(clip.tokenize(ins_2, truncate=True).cuda()).float().data.cpu().numpy()).cuda()]
            
            out_11, out_12, out_21, out_22 = G(con_1, ins_1), G(con_1, ins_2), G(con_2, ins_1), G(con_2, ins_2)
            o4_11, o4_12, o4_21, o4_22 = G.enc_c4(out_11), G.enc_c4(out_12), G.enc_c4(out_21), G.enc_c4(out_22)
            o5_11, o5_12, o5_21, o5_22 = G.enc_c5(o4_11), G.enc_c5(o4_12), G.enc_c5(o4_21), G.enc_c5(o4_22)
            ls_cc = (loss_mse(o4_11, o4_12)+loss_mse(o4_21, o4_22)+loss_mse(o5_11, o5_12)+loss_mse(o5_21, o5_22))/4.0
            ls_cs = (loss_percept(o4_11, o4_21)+loss_percept(o4_12, o4_22)+loss_percept(o5_11, o5_21)+loss_percept(o5_12, o5_22))/4.0
            
            optzr.zero_grad(), (ls_cc+ls_cs).backward(), optzr.step()
            
            ls_cc, ls_cs = ls_cc.item(), ls_cs.item()
            bh_cc.append(ls_cc), bh_cs.append(ls_cs)
            TQ.set_postfix(ls_cc='%.3f'%(ls_cc), ls_cs='%.3f'%(ls_cs))
            
            step += 1
            if step%500==0:
                bh_cc, bh_cs = np.mean(bh_cc).item(), np.mean(bh_cs).item()
                print('Step %d: ls_cc=%.4f / ls_cs=%.4f'%(step, bh_cc, bh_cs))
                bh_cc, bh_cs, = [], []
                T.save(G.state_dict(), './_ckpt/ctr.pt')
                
