import random
import sys

"""
take VOC2007's trainval.txt from the segmentation set, select 300 images, and put 
these into tudtrain$NUM.txt, motorbike_tudtrain$NUM.txt
and background_tudtrain$NUM.txt.

python randomize_tud_bg.py

"""

def randomize_tud_bg(appendix):
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

def randomize_trainsets(appendix):
    classes =['aeroplane','bicycle','bird','boat','bottle','bus','car','cat',\
        'chair','cow','diningtable','dog','horse','motorbike','person',\
        'pottedplant','sheep','sofa','train','tvmonitor']
    base_trainset = 'VOCdevkit/VOC2007/ImageSets/Main/train.txt'
    base_clstrainset = 'VOCdevkit/VOC2007/ImageSets/Main/%s_train.txt'
    output_trainset = 'VOCdevkit/VOC2007/ImageSets/Main/train%s.txt'
    output_clstrainset = 'VOCdevkit/VOC2007/ImageSets/Main/%s_train%s.txt'
    
    seed = random.random()
    
    btrf = open(base_trainset, 'r')
    
    data = btrf.read().split('\n')
    btrf.close()
    random.seed(seed)
    random.shuffle(data)
    otrf = open(output_trainset%appendix, 'w')
    otrf.write('\n'.join(data))
    otrf.close()
    
    for cls in classes:
        btrf = open(base_clstrainset%cls, 'r')
        data = btrf.read().split('\n')
        btrf.close()
        random.seed(seed)
        random.shuffle(data)
        otrf = open(output_clstrainset%(cls,appendix), 'w')
        otrf.write('\n'.join(data))
        otrf.close()

if __name__ == "__main__":
    randomize_trainsets(sys.argv[1])
    # randomize_tud_bg(sys.argv[1])
