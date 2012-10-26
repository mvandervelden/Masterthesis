import Image

ann_str = """<annotation>
	<folder>VOC2007</folder>
	<filename>%s</filename>
	<source>
		<database>The TUD motorbike Database</database>
		<annotation>TUD Motorbike</annotation>
		<image>flickr</image>
		<flickrid>000000</flickrid>
	</source>
	<owner>
		<flickrid>000000</flickrid>
		<name>?</name>
	</owner>
	<size>
		<width>%d</width>
		<height>%d</height>
		<depth>3</depth>
	</size>
	<segmented>%d</segmented>
%s
</annotation>"""

obj_str = """	<object>
		<name>motorbike</name>
		<pose>?</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>%d</xmin>
			<ymin>%d</ymin>
			<xmax>%d</xmax>
			<ymax>%d</ymax>
		</bndbox>
	</object>
"""

def annotate_test(teim):
    for im in teim:
        print im
        for l in idl_str:
            if im in l:
                ann = l
                break
        bbs = ann.split(': ')[1][:-2]
        imstr = ""
        bbs = eval('['+bbs+']')
        for bb in bbs:
            if bb[0] < bb[2]:
                xmin = bb[0]
                ymax = bb[2]
            else:
                xmin = bb[2]
                xmax = bb[0]
            if bb[1] < bb[3]:
                ymin = bb[1]
                ymax = bb[3]
            else:
                ymin = bb[3]
                ymax = bb[1]
            print imstr
            print xmin, ymin, xmax, ymax
            imstr += obj_str%(xmin, ymin, xmax, ymax)
        print imstr
        ii = Image.open('../../JPEGImages/%s.png'%im)
    
        with open('../../Annotations/%s.xml'%im, 'w') as xf:
            xf.write(ann_str%(im+'.png', ii.size[0], ii.size[1], 0, imstr))

def annotate_train(imset):
    for im in imset:
        print im
        msk = Image.open('TUDmotorbikes/SegmentationClass/%s.png'%im)
        bb = msk.getbbox()
        print 'bb: ', str(bb)
        imstr = obj_str%(bb[0],bb[1],bb[2], bb[3])
        with open('TUDmotorbikes/Annotations/%s.xml'%im, 'w') as xf:
            xf.write(ann_str%(im+'.png', msk.size[0], msk.size[1], 1, imstr))


def convert_masks(imset):
    example_mask = Image.open('VOCdevkit/VOC2007/SegmentationClass/000032.png')
    palette = example_mask.getpalette()
    for im in imset:
        oldmask = Image.open('TUDmotorbikes/Masks/%s.png'%im)
        bwmask = oldmask.convert('L')
        thresholdmask = Image.eval(bwmask, lambda a: 0 if a <200 else 255)
        newmask = Image.new('P', oldmask.size)
        newmask.putpalette(palette)
        newmask.paste(14, thresholdmask)
        newmask.save('TUDmotorbikes/SegmentationClass/%s.png'%im)

def load_imset(imsetfile):
    imset = []
    with open(imsetfile,'r') as ff:
        for line in ff:
            imid = line.strip()
            if len(imid) == 4:
                imset.append(imid)
    return imset

if __name__ == '__main__':
    imsetfile = 'TUDmotorbikes/ImageSets/Main/tudtrain.txt'
    imset = load_imset(imsetfile)
    # annotate_test()
    # convert_masks()
    annotate_train(imset)