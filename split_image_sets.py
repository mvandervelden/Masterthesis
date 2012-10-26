from nbnn import voc
import random
import sys

"""
Split an image set randomly into two sets while keeping the class distributions
mostly equal. If RATIO is not specified both sets will have about half of the
original set, else the first set will have RATIO part of the image set and the
second set will have 1-RATIO of the image set.

python split_image_set.py IM_SET VOC_CONFIG OUTSET1 OUTSET2 [RATIO]

"""


def split_clsfile(infile, outfile1, ims1, outfile2=None, ims2 = None):
    if not ims2 is None:
        split = True
    else:
        split = False
         
    imids1 = [im.im_id for im in ims1]
    if split:
        imids2 = [im.im_id for im in ims2]
    with open(infile, 'r') as f:
        content = f.read().split('\n')
    filestr1 = []
    if split:
        filestr2 = []
    for line in content:
        imid = line.split(' ')[0]
        if imid in imids1:
            filestr1.append(line)
        elif split and imid in imids2:
            filestr2.append(line)
    no_pos1 = len([l for l in filestr1 if int(l.split()[1]) == 1])
    no_neg1 = len([l for l in filestr1 if int(l.split()[1]) == -1])
    no_zer1 = len([l for l in filestr1 if int(l.split()[1]) == 0])
    print outfile1+": %d (%d/%d/%d)"%(len(filestr1), no_pos1, no_neg1, no_zer1)
    with open(outfile1, 'w') as of1:
        for line in filestr1:
            of1.write(line+'\n')
    
    if split:
        no_pos2 = len([l for l in filestr2 if int(l.split()[1]) == 1])
        no_neg2 = len([l for l in filestr2 if int(l.split()[1]) == -1])
        no_zer2 = len([l for l in filestr2 if int(l.split()[1]) == 0])
        print outfile2+": %d (%d/%d/%d)"%(len(filestr2), no_pos2, no_neg2, no_zer2)
        with open(outfile2, 'w') as of2:
            for line in filestr2:
                of2.write(line+'\n')

# if __name__ == '__main__':
#     im_set_cls = sys.argv[1]
#     im_set_src = sys.argv[2]
#     voc_cfg = sys.argv[3]
#     
#     VOCopts = voc.VOC.fromConfig(voc_cfg)
#     classes = [c for c in VOCopts.classes if not c == "background"]
#     ims = voc.read_image_set(VOCopts, im_set_src)
#     for cls in classes:
#         split_clsfile(VOCopts.imset_path%(cls+'_'+im_set_cls), \
#             VOCopts.imset_path%(cls+'_'+im_set_src), ims)
if __name__ == "__main__":

    im_set = sys.argv[1]
    voc_cfg = sys.argv[2]

    outset1 = sys.argv[3]
    outset2 = sys.argv[4]

    ratio = 0.5
    if len(sys.argv) > 5:
        ratio = float(sys.argv[5])
    dset = 'voc'
    if len(sys.argv) > 6:
        dset = sys.argv[6]
    
    VOCopts = voc.VOC.fromConfig(voc_cfg)
    if dset == 'tud':
        classes = [c for c in VOCopts.classes]
    else:
        classes = [c for c in VOCopts.classes if not c == "background"]
    ims = voc.read_image_set(VOCopts, im_set)

    im_set1 = set()
    im_set2 = set()
    if dset == 'voc':
        for class_name in classes:
            class_ims = [im for im in ims if class_name in im.classes]
            random.shuffle(class_ims)
            idx = int(round(len(class_ims)*ratio))
            im_set1.update(set(class_ims[:idx]).difference(im_set2))
            im_set2.update(set(class_ims[idx:]).difference(im_set1))
    elif dset == 'tud':
        class_name = 'motorbike'
        class_ims = [im for im in ims if len(im.im_id) == 4]
        random.shuffle(class_ims)
        idx = int(round(len(class_ims)*ratio))
        im_set1.update(set(class_ims[:idx]).difference(im_set2))
        im_set2.update(set(class_ims[idx:]).difference(im_set1))
        
        class_name = 'background'
        class_ims = [im for im in ims if len(im.im_id) == 6]
        random.shuffle(class_ims)
        idx = int(round(len(class_ims)*ratio))
        im_set1.update(set(class_ims[:idx]).difference(im_set2))
        im_set2.update(set(class_ims[idx:]).difference(im_set1))
        
    with open(VOCopts.imset_path%outset1, 'w') as f:
        for im in im_set1:
            f.write(im.im_id + "\n")

    with open(VOCopts.imset_path%outset2, 'w') as f:
        for im in im_set2:
            f.write(im.im_id + "\n")

    print outset1, "length:", len(im_set1)
    print outset2, "length:", len(im_set2)
    for class_name in classes:
        s1 = len([im for im in im_set1 if class_name in im.classes])
        s2 = len([im for im in im_set2 if class_name in im.classes])
        print
        print class_name
        print outset1+":", s1
        print outset2+":", s2
        
        split_clsfile(VOCopts.imset_path%(class_name + '_' + im_set), \
            VOCopts.imset_path%(class_name + '_' + outset1), im_set1, \
            VOCopts.imset_path%(class_name + '_' + outset2), im_set2)

