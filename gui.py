
import sys
import PyQt5.QtWidgets, PyQt5.QtGui

import numpy as np
import torch as T
import torchvision as TV
import clip, model

from PIL import Image, ImageQt

class GUI(PyQt5.QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.clip, _ = clip.load('ViT-B/32', device='cuda')
        self.clva = model.CLVA(512).cuda().eval()
        self.clva.load_state_dict(T.load('./_ckpt/clva_dtd.pt', map_location='cpu'))

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('CLVA for LDAST')
        self.move(400, 400), self.setFixedSize(50+256+50+512+50+512+50, 50+384+50)

        self.bar_sta = self.statusBar()
        self.bar_sta.showMessage('CLVA Ready')

        self.btn_load = PyQt5.QtWidgets.QPushButton('Load', self)
        self.btn_load.move(50, 50), self.btn_load.setFixedSize(120, 50), self.btn_load.setFont(PyQt5.QtGui.QFont('Arial', 10))
        self.btn_load.clicked.connect(self.press_load)

        self.btn_run = PyQt5.QtWidgets.QPushButton('Run', self)
        self.btn_run.move(50+120+16, 50), self.btn_run.setFixedSize(120, 50), self.btn_run.setFont(PyQt5.QtGui.QFont('Arial', 10))
        self.btn_run.clicked.connect(self.press_run)

        self.ins = PyQt5.QtWidgets.QPlainTextEdit(self)
        self.ins.move(50, 50+50+25), self.ins.setFixedSize(256, 284), self.ins.setFont(PyQt5.QtGui.QFont('Arial', 10))

        self.img_inp = PyQt5.QtWidgets.QLabel(self)
        self.img_inp.move(50+256+50, 50), self.img_inp.setFixedSize(512, 384)

        self.img_out = PyQt5.QtWidgets.QLabel(self)
        self.img_out.move(50+256+50+512+50, 50), self.img_out.setFixedSize(512, 384)

        self.show()

    def press_load(self):
        self.file = PyQt5.QtWidgets.QFileDialog.getOpenFileName()[0]

        if self.file=='': return
        else: self.img_inp.setPixmap(PyQt5.QtGui.QPixmap(self.file).scaled(512, 384))

        self.bar_sta.showMessage('Read %s'%(self.file))

    def press_run(self):
        with T.no_grad():
            x = self.ins.toPlainText()
            x = T.from_numpy(self.clip.encode_text(clip.tokenize([x], truncate=True).cuda()).float().data.cpu().numpy())
            c = Image.open(self.file).convert('RGB')
            c = TV.transforms.ToTensor()(c).unsqueeze(0)
            o = self.clva(c.cuda(), x.cuda())

            o = (o[0].permute(1, 2, 0).data.cpu().numpy().clip(0.0, 1.0)*255.0).astype(np.uint8)
            o = Image.fromarray(o).convert('RGB')
            o.save('out.jpg', quality=100, subsampling=0)
            self.img_out.setPixmap(PyQt5.QtGui.QPixmap('out.jpg').scaled(512, 384))

        self.bar_sta.showMessage('Run %s and save as out.jpg'%(self.file))

if __name__=='__main__':
    app = PyQt5.QtWidgets.QApplication(sys.argv)
    gui = GUI()
    sys.exit(app.exec_())
    
