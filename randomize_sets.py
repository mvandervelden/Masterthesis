import random
import sys, os

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

def randomize_graz01_sets(appendix):
    bike_folder = 'Graz01/bikes'
    person_folder = 'Graz01/persons'
    bg_folder = 'Graz01/no_bike_no_person'
    image_set_folder = 'Graz01/ImageSets/%s_%s%s.txt'
    # We make: person_persontrain##.txt, bg_persontrain##.txt, person_persontest##.txt, bg_persontest##.txt, bike_biketrain##.txt, bg_biketrain##.txt, bike_biketest##.txt, bg_biketest##.txt
    
    all_bikes = os.listdir(bike_folder)
    all_persons = os.listdir(person_folder)
    all_bg = os.listdir(bg_folder)
    
    # shuffle
    random.shuffle(all_bikes)
    random.shuffle(all_persons)
    random.shuffle(all_bg)
    
    # select:
    p_persontrain = all_persons[:100]
    p_persontest = all_persons[100:200]
    bg_persontrain = all_bikes[:50] + all_bg[:50]
    bg_persontest = all_bikes[50:100] + all_bg[50:100]
    b_biketrain = all_bikes[:100]
    b_biketest = all_bikes[100:200]
    bg_biketrain = all_persons[:50] + all_bg[:50]
    bg_biketest = all_persons[50:100] + all_bg[50:100]
    with open(image_set_folder%('person', 'persontrain',appendix), 'w') as ptf:
        with open(image_set_folder%('bg', 'persontrain',appendix), 'w') as btf:
            for p in range(100):
                ptf.write('%s  1\n'%p_persontrain[p][:-4])
                btf.write('%s -1\n'%p_persontrain[p][:-4])
            for p in range(100):
                ptf.write('%s -1\n'%bg_persontrain[p][:-4])
                btf.write('%s  1\n'%bg_persontrain[p][:-4])
    with open(image_set_folder%('person', 'persontest',appendix), 'w') as ptf:
        with open(image_set_folder%('bg', 'persontest',appendix), 'w') as btf:
            for p in range(100):
                ptf.write('%s  1\n'%p_persontest[p][:-4])
                btf.write('%s -1\n'%p_persontest[p][:-4])
            for p in range(100):
                ptf.write('%s -1\n'%bg_persontest[p][:-4])
                btf.write('%s  1\n'%bg_persontest[p][:-4])
    with open(image_set_folder%('bike', 'biketrain',appendix), 'w') as ptf:
        with open(image_set_folder%('bg', 'biketrain',appendix), 'w') as btf:
            for p in range(100):
                ptf.write('%s  1\n'%b_biketrain[p][:-4])
                btf.write('%s -1\n'%b_biketrain[p][:-4])
            for p in range(100):
                ptf.write('%s -1\n'%bg_biketrain[p][:-4])
                btf.write('%s  1\n'%bg_biketrain[p][:-4])
    with open(image_set_folder%('bike', 'biketest',appendix), 'w') as ptf:
        with open(image_set_folder%('bg', 'biketest',appendix), 'w') as btf:
            for p in range(100):
                ptf.write('%s  1\n'%b_biketest[p][:-4])
                btf.write('%s -1\n'%b_biketest[p][:-4])
            for p in range(100):
                ptf.write('%s -1\n'%bg_biketest[p][:-4])
                btf.write('%s  1\n'%bg_biketest[p][:-4])
    

if __name__ == "__main__":
    randomize_graz01_sets(sys.argv[1])
    # randomize_trainsets(sys.argv[1])
    # randomize_tud_bg(sys.argv[1])
