from PIL import Image
import os
import numpy as np
infolder = 'scratchdisk/im/caltech101/101_ObjectCategories'
outfolder = 'scratchdisk/im/caltechresize/101_ObjectCategories'
cats = os.listdir(infolder)
os.mkdir('scratchdisk/im/caltechresize')
os.mkdir(outfolder)
for cat in cats:
    ims = os.listdir('/'.join([infolder,cat]))
    os.mkdir('/'.join([outfolder,cat]))
    for im in ims:
        img = Image.open('/'.join([infolder,cat,im]))
        # Get the longest side
        max_d = np.argmax(img.size)
        #Set the longest side to be 300 pixels, scale the short side accordingly
        if max_d == 0:
            width = 300
            height = int(img.size[1]*float(width)/img.size[0])
        else:
            height = 300
            width = int(img.size[0]*float(height)/img.size[1])
        resized_img=img.resize((width,height),Image.ANTIALIAS)
        resized_img.save('/'.join([outfolder,cat,im]))
        print 'resized '+cat+'/'+im