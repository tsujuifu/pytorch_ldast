
from tqdm import tqdm

import numpy as np
import torch as T
import torchvision as TV

from PIL import Image

import clip
import model

if __name__=='__main__':
    CLIP, _ = clip.load('ViT-B/32', device='cuda')
    
    clva = model.CLVA(512).cuda().eval()
    clva.load_state_dict(T.load('./_ckpt/clva_dtd.pt', map_location='cpu'))

    for i, x in tqdm(enumerate(['yellow, teal, gold, dimpled, protuberant lines and dappled dots and loops on smoother areas, shapes like large commas', 
                                'stained with green and red paint bit of a dust as well, looks like a cleaning cloth', 
                                'silver meshed metal, twisted, spirals, with a smooth white background surface', 
                                'a road which is different color like red and small blue marble colors', 
                                'white swirly design on blue background, soft texture', 
                                'spiralled red yellow shiny concentric', 
                                'patches of orange yellow and red mixed', 
                                'green, white, smooth, repeating, bumpy', 
                                'tiger,skin,black stripes,yellow color,hairs', 
                                'black white dirty parallel swirly', 
                                'colorful, abstract shapes, damp, thick, black splatter', 
                                'crystal, white, water color, green, grey', 
                                'blue background, yellowish, crosshatched, vertical, horizontally and diagonally designed', 
                                'frilly, green, red, vein, fading color']), ascii=True):
        with T.no_grad():
            c = Image.open('./_content/%d.jpg'%(i)).convert('RGB')
            c = TV.transforms.ToTensor()(c).unsqueeze(0)
            x = T.from_numpy(CLIP.encode_text(clip.tokenize([x], truncate=True).cuda()).float().data.cpu().numpy())
            o = clva(c.cuda(), x.cuda())
            
            o = (o[0].permute(1, 2, 0).data.cpu().numpy().clip(0.0, 1.0)*255.0).astype(np.uint8)
            o = Image.fromarray(o).convert('RGB').save('./_output/%d.jpg'%(i), quality=100)
            