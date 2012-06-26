from ConfigParser import RawConfigParser
import os, logging
from random import shuffle

log = logging.getLogger(__name__)

class Caltech(object):
    def __init__(self, classes, image_path):
        self.classes = classes
        self.image_path = image_path
        self.cls_path = '/'.join(image_path.split('/')[:-1])

    @classmethod
    def fromConfig(cls,config):
        cfg = RawConfigParser()
        cfg.read(config)

        classes = cfg.get("CAL","classes").split()
        image_path = cfg.get("CAL","image_path")

        return Caltech(classes, image_path)

class CalImage(object):
    def __init__(self, path, im_id, cls):
        self.path = path
        self.im_id = im_id
        self.cls = cls
        self.descriptor_file = []

def read_cal_image_set(CALopts, cls, train_size, test_size):
    cls_folder = CALopts.cls_path%cls
    images = os.listdir(cls_folder)
    shuffle(images)
    no_files = train_size + test_size
    train_files = []
    for i in range(train_size):
        im_path ='/'.join([cls_folder, images[i]])
        im_id = cls + '__' + images[i].rstrip('.jpg')
        train_files.append(CalImage(im_path, im_id, cls))
    
    test_files = []
    if no_files > len(images):
        log.debug("Too little images in class %s, padding %d files", cls, no_files-len(images))
        mxfile = len(images)
    else:
        mxfile = no_files
    for i in range(train_size, mxfile):
        im_path ='/'.join([cls_folder, images[i]])
        im_id = cls + '__' + images[i].rstrip('.jpg')
        test_files.append(CalImage(im_path, im_id, cls))
    for i in range(mxfile, no_files):
        log.debug('-- Padding with %d',i-mxfile)
        test_files.append(test_files[i-mxfile])
    return train_files, test_files
