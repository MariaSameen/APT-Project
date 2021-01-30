import glob
import numpy as np
from PIL import Image

w, h = 64, 64
i = 0

for filename in glob.iglob('C:\\Users\\Maria Sameen\\Desktop\\apt_project\\normal_dataset\\normal_exe\\*.exe'):
    print(filename)
    with open(filename, mode='rb') as f:
        d = np.fromfile(f, dtype=np.uint8, count=w*h).reshape(h, w)
    PILimage = Image.fromarray(d)
    i = i+1
    PILimage.save('C:\\Users\\Maria Sameen\\Desktop\\apt_project\\normal_dataset\\normal_image\\'+str(i)+'.png')
