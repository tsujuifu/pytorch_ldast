
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
        nc, ns = idx//self.N_STYLE, idx%self.N_STYLE
        
        con = TV.transforms.ToTensor()(b2f(self.pkl['content'][nc]))
        sty = TV.transforms.ToTensor()(b2f(self.pkl['style'][ns]['image']))
        ins = np.random.choice(self.pkl['style'][ns]['instruction'])
        
        return con, sty, ins

def get_patch(img):
    ret = []
    for b in range(img.shape[0]):
        for _ in range(4):
            x, y = np.random.randint(0, img.shape[2]-32), np.random.randint(0, img.shape[3]-32)
            ret.append(img[b, :, x:x+32, y:y+32].unsqueeze(0))
    ret = T.cat(ret, dim=0)
    return ret

def loss_percept(f1, f2):
    [B, C, H1, W1], [_, _, H2, W2] = f1.shape, f2.shape
    f1, f2 = f1.view([B*C, H1*W1]), f2.view([B*C, H2*W2])
    f1, f2 = T.mm(f1, f1.t())/(H1*W1), T.mm(f2, f2.t())/(H2*W2)
    ret = T.nn.functional.mse_loss(f1, f2)
    return ret
    
if __name__=='__main__':
    dl = T.utils.data.DataLoader(DS('./_data/dtd.pkl'), batch_size=24, num_workers=12, shuffle=True, pin_memory=True)
    
    CLIP, _ = clip.load('ViT-B/32', device='cuda')
    G, D = CLVA(512).cuda().train(), Discriminator().cuda().train()
    G.load_state_dict(T.load('./_ckpt/sanet.pt', map_location='cpu'), strict=False)
    
    loss_mse, loss_bce = T.nn.MSELoss().cuda(), T.nn.BCELoss().cuda()
    optzr_g, optzr_d = [T.optim.AdamW(G.parameters(), lr=0.0003, betas=(0.9, 0.98)), 
                        T.optim.AdamW(D.parameters(), lr=0.0001, betas=(0.9, 0.98))]
    print('G:', [(n, p.requires_grad) for n, p in G.named_parameters()])
    print('D:', [(n, p.requires_grad) for n, p in D.named_parameters()])
    
    step, TQ = 0, tqdm(dl, ascii=True)
    while True:
        bh_rec, bh_psd, bh_cm, bh_sm = [], [], [], []
        for con, sty, ins in TQ:
            con, sty = con.cuda(), sty.cuda()
            ins = T.from_numpy(CLIP.encode_text(clip.tokenize(ins, truncate=True).cuda()).float().data.cpu().numpy()).cuda()
            
            rec, out = G.forward_cs(con, con), G.forward(con, ins)
            ls_rec = loss_mse(rec, con)
            
            po, ins = get_patch(out), ins.unsqueeze(1).expand(-1, 4, -1).flatten(0, 1)
            dpo = D(po, ins)
            ls_psd = loss_bce(dpo, T.ones(dpo.shape).cuda())
            
            c4, s4 = G.enc_c4(con), G.enc_c4(sty)
            c5, s5 = G.enc_c5(c4), G.enc_c5(s4)
            o4 = G.enc_c4(out)
            o5 = G.enc_c5(o4)
            ls_cm, ls_sm = [(loss_mse(o4, c4)+loss_mse(o5, c5))/2.0, 
                            (loss_percept(o4, s4)+loss_percept(o5, s5))/2.0]
            
            optzr_g.zero_grad(), (ls_rec+ls_psd+ls_cm+ls_sm).backward(), optzr_g.step()
            
            out = T.from_numpy(out.data.cpu().numpy()).cuda()
            po, ps = get_patch(out), get_patch(sty)
            dpo, dps = D(po, ins), D(ps, ins)
            ls_dsc = (loss_bce(dpo, T.zeros(dpo.shape).cuda())+loss_bce(dps, T.ones(dps.shape).cuda()))/2.0
            
            optzr_d.zero_grad(), (ls_dsc).backward(), optzr_d.step()
            
            ls_rec, ls_psd, ls_cm, ls_sm = ls_rec.item(), ls_psd.item(), ls_cm.item(), ls_sm.item()
            bh_rec.append(ls_rec), bh_psd.append(ls_psd), bh_cm.append(ls_cm), bh_sm.append(ls_sm)
            TQ.set_postfix(ls_rec='%.3f'%(ls_rec), ls_psd='%.3f'%(ls_psd), ls_cm='%.3f'%(ls_cm), ls_sm='%.3f'%(ls_sm))
            
            step += 1
            if step%500==0:
                bh_rec, bh_psd, bh_cm, bh_sm = np.mean(bh_rec).item(), np.mean(bh_psd).item(), np.mean(bh_cm).item(), np.mean(bh_sm).item()
                print('Step %d: ls_rec=%.4f / ls_psd=%.4f / ls_cm=%.4f / ls_sm=%.4f'%(step, bh_rec, bh_psd, bh_cm, bh_sm))
                bh_rec, bh_psd, bh_cm, bh_sm = [], [], [], []
                T.save(G.state_dict(), './_ckpt/lva.pt')
                
