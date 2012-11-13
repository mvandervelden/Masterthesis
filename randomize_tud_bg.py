import random
import sys

"""
take VOC2007's trainval.txt from the segmentation set, select 300 images, and put 
these into tudtrain$NUM.txt, motorbike_tudtrain$NUM.txt
and background_tudtrain$NUM.txt.

python randomize_tud_bg.py

"""

if __name__ == "__main__":
    
    appendix = sys.argv[1]
    bg_ims = 300
    base_bg_set = 'VOCdevkit/VOC2007/ImageSets/Segmentation/trainval.txt'
    base_fg_set = 'VOCdevkit/VOC2007/ImageSets/Main/tudtrain_fg.txt'
    
    output_train_set = 'VOCdevkit/VOC2007/ImageSets/Main/tudtrain%s.txt'%appendix
    output_mb_set    = 'VOCdevkit/VOC2007/ImageSets/Main/motorbike_tudtrain%s.txt'%appendix
    output_bg_set    = 'VOCdevkit/VOC2007/ImageSets/Main/background_tudtrain%s.txt'%appendix
    
    imset_mb = []
    imset_bg = []
    
    for line in open(base_fg_set, 'r'):
        imset_mb.append(line.strip())
    
    for line in open(base_bg_set, 'r'):
        imset_bg.append(line.strip())
    
    random.shuffle(imset_bg)
    imset_bg = imset_bg[:300]
    
    with open(output_train_set, 'w') as trainfile:
        with open(output_mb_set, 'w') as motorbikefile:
            with open(output_bg_set, 'w') as backgroundfile:
                for i, im in enumerate(imset_mb):
                    print 'fg: %3d: %s'%(i, im)
                    trainfile.write(im+'\n')
                    motorbikefile.write(im+'  1\n')
                    backgroundfile.write(im+' -1\n')
                for i, im in enumerate(imset_bg):
                    print 'bg: %3d: %s'%(i, im)
                    trainfile.write(im+'\n')
                    motorbikefile.write(im+' -1\n')
                    backgroundfile.write(im+' 1\n')

