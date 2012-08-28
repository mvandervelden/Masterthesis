from nbnn.vocimage import *
from nbnn.voc import *
import sys

if __name__ == "__main__":
    imset = sys.argv[1]
    configfile = sys.argv[2]
    if len(sys.argv) > 3:
        print_classes = True
    else:
        print_classes = False
    classes =['aeroplane','bird','motorbike','person']
    VOCopts = VOC.fromConfig(configfile)
    
    images = read_image_set(VOCopts, imset)
    
    selection = dict()
    for cls in classes:
        selection[cls] = []
    
    
    for image in images:
        if len(image.objects) == 1:
            obj = image.objects[0]
            obcls = obj.class_name
            
            if obcls in classes and len(selection[obcls]) < 5:
                if not obj.difficult and not obj.truncated:
                    objwidth = obj.xmax-obj.xmin
                    objheight = obj.ymax-obj.ymin
                    if objwidth <= 0.5*image.width and objwidth >= 0.20*image.width and \
                            objheight <= 0.5*image.height and objheight >= 0.20*image.height:
                        selection[obcls].append(image.im_id)
    
    for cls, im in selection.items():
        if not print_classes:
            for imm in im:
                print imm
        else:
            for imm in im:
                print "%11s %11s"%(cls, imm)