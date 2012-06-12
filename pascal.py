from nbnn import *
import xml.parsers.expat as expat
import re,os,math
import numpy.random as rndm

log = logging.getLogger(__name__)
class VOCDetection(VOC):
    
    def __init__(self, classes, trainset_file, testset_file, image_path,
            gt_annotation_path):
        log.info("Creating VOCDetection.")
        super(VOCDetection,self).__init__(classes, trainset_file, \
                testset_file, image_path)
        self.gt_annotation_path = gt_annotation_path
        self.training = True
        
    def get_segmentation(self, im_path):
        """Load annotation file of an image, and parse the xml to find all
        bounding boxes within the file
        
        Keyword arguments:
        im_path - path to an image
        
        Returns: a MultipleSegmentation of the list of bounding boxes in the
        image
        
        """
        
        im_id = self.get_im_id(im_path)
        annotation_file = self.gt_annotation_path%im_id
        objects = self.get_objects_from_xml(annotation_file)
        if self.training:
            bboxes = [o['bbox'] for o in objects if o['name'] in self.classes]
        else:
            bboxes = [o['bbox'] for o in objects]
        return MultipleSegmentation(bboxes)
    
    def get_ground_truth(self, im_path, segmentation):
        """Get the class of all objects in an image, using its annotation file
        
        Keyword arguments:
        im_path - the filename and path of an image, used to get its ID
        segmentation - Segmentation object of the image, not needed.
        
        Returns: list of class strings.
        
        """
        
        im_id = self.get_im_id(im_path)
        annotation_file = self.gt_annotation_path%im_id
        objects = self.get_objects_from_xml(annotation_file)
        if self.training:
            ground_truth = [o['name'] for o in objects if o['name'] in self.classes]
        else:
            ground_truth = [o['name'] for o in objects]
        return ground_truth
    
    def get_objects_from_xml(self,annotation_file):
        """Parse an xml to get the objects containing class names, annotations,
        and bounding boxes of an image from the annotation file
        
        Keyword arguments:
        annotation_file - path to an annotation file
        
        Returns: A list of object dicts, each having keys
            {name:str, pose:str, truncated:bool, difficult:bool, bbox:list},
            where bbox is represented as [xmin,ymin,xmax,ymax]
        
        """
        
        class XParser:
            """Class with parsing functions for the xml, that keeps track of the
            the objects in the file.
            
            """
            
            def __init__(self,difficult=False):
                """Initialize the XParser object by initializing variables.
                Difficult objects are ignored by default
                
                """
                
                self.bboxlist = ['xmin','ymin','xmax','ymax']
                self.depth = -1
                self.cur_id = 0
                self.cur_field = None
                self.cur_object = None
                self.objects = []
                self.s = re.compile('[a-zA-Z0-9]+')
                self.difficult = difficult
            
            def start_element(self,name, attrs):
                """ Callback function when the expat parser finds a start
                element. Keeps track of how deep we are in the xml structure,
                and which field is the current one.
                
                Keyword arguments:
                name - element name (str)
                attrs - possible element attributes (non-existent in VOC)
                
                """
                # Increase the depth in the xml we are in
                self.depth += 1
                # Set the current field (e.g. name,bndbox,object,difficult)
                self.cur_field = name
                if self.depth == 1 and name == 'object':
                    # If we encounter an object, start a dict for it
                    self.cur_object = dict()
                elif self.depth == 2 and name == 'bndbox':
                    # If we encounter a bounding box, initialize it as a list
                    self.cur_object['bbox'] = [0]*4
                
            def end_element(self,name):
                """ Calback function when the expat parser finds the end of an
                element. Keeps track of how deep we are in the xml structure,
                and stores a completely object in the object list if needed.
                
                Keyword arguments:
                name - the name of the element that is ended.
                
                """
                # Decrease depth in the xml
                self.depth -= 1
                if self.depth == 0 and name == 'object':
                    # If we step out of an object, reset the current one and
                    # append it to the list of objects if it is not difficult
                    # (unless we decide to include difficult objects
                    # ,self.difficult = True in that case
                    if self.difficult or not self.cur_object['difficult']:
                        self.objects.append(self.cur_object)
                    self.cur_object = None
                    
            def char_data(self,data):
                """Callback function when the expat parser finds character data
                within an element. Stores the values found in the current object
                dict, when this is applicable.
                
                Keyword arguments:
                data - string that the parser found.
                
                """
                # remove possible extra spaces or newlines
                data = data.strip()

                if not self.cur_object is None and not data == '':
                    # If we have an object, and there is data (sometimes empty
                    # lines are parsed in this function)
                    if self.depth == 2:
                        # On the first object level, we parse name (class) and
                        # difficulty
                        if self.cur_field == 'name':
                            # set name (self.cur_field) to the class (data)
                            self.cur_object[self.cur_field] = data
                        elif self.cur_field == 'difficult':
                            # Set the difficulty: if it is '1', evaluate to True
                            # else, evaluate to False
                            self.cur_object[self.cur_field] = data == '1'
                    elif self.depth == 3 and \
                        self.cur_field in ['xmin','ymin','xmax','ymax']:
                        # if we are at level 3 and have a bounding box item
                        # (one field of [xmin,ymin,xmax,ymax]), set the current
                        # value (data) of the bbox-coordinate (self.curfield)
                        # get the correct list index of the bbox-coordinate
                        idx = self.bboxlist.index(self.cur_field)
                        self.cur_object['bbox'][idx] = int(data)
            
            def get_objects(self):
                """Getter that returns the list of objects.
                
                """
                return self.objects
        # Create a parser
        p = expat.ParserCreate()
        xp = XParser()
        # Add element handlers
        p.StartElementHandler = xp.start_element
        p.EndElementHandler = xp.end_element
        p.CharacterDataHandler = xp.char_data
        with open(annotation_file,'r') as af:
            # Parse the file
            p.ParseFile(af)
        # Get the objects encountered
        return xp.get_objects()
        
    def toggle_training(self):
        self.training = not self.training
        return self.training

class VOCClassification(VOCDetection):
    
    def get_segmentation(self, im_path):
        """Load annotation file of an image, and parse the xml to find all
        bounding boxes within the file
        
        Keyword arguments:
        im_path - path to an image
        
        Returns: a MultipleSegmentation of the list of bounding boxes in the
        image
        
        """

        if self.training:
            im_id = self.get_im_id(im_path)
            annotation_file = self.gt_annotation_path%im_id
            objects = self.get_objects_from_xml(annotation_file)
            ground_truth = set([o['name'] for o in objects if o['name'] in self.classes])
            return Classification(len(ground_truth))
        else:
            return Classification(1)
    
    def get_ground_truth(self, im_path, segmentation):
        """Get the class of all objects in an image, using its annotation file
        
        Keyword arguments:
        im_path - the filename and path of an image, used to get its ID
        segmentation - Segmentation object of the image, not needed.
        
        Returns: list of class strings.
        
        """
        
        im_id = self.get_im_id(im_path)
        annotation_file = self.gt_annotation_path%im_id
        objects = self.get_objects_from_xml(annotation_file)
        if self.training:
            ground_truth = list(set([o['name'] for o in objects if o['name'] in self.classes]))
        else:
            ground_truth = list(set([o['name'] for o in objects]))
        return ground_truth

    
class CaltechClassification(Dataset):
    
    def __init__(self, image_path,trainsize=20,testsize=15, no_classes=None):
        classes = ["BACKGROUND_Google","Faces","Faces_easy","Leopards",\
            "Motorbikes","accordion","airplanes","anchor","ant","barrel",\
            "bass","beaver","binocular","bonsai","brain","brontosaurus",\
            "buddha","butterfly","camera","cannon","car_side","ceiling_fan",\
            "cellphone","chair","chandelier","cougar_body","cougar_face","crab",\
            "crayfish","crocodile","crocodile_head","cup","dalmatian",\
            "dollar_bill","dolphin","dragonfly","electric_guitar","elephant",\
            "emu","euphonium","ewer","ferry","flamingo","flamingo_head",\
            "garfield","gerenuk","gramophone","grand_piano","hawksbill",\
            "headphone","hedgehog","helicopter","ibis","inline_skate",\
            "joshua_tree","kangaroo","ketch","lamp","laptop","llama","lobster",\
            "lotus","mandolin","mayfly","menorah","metronome","minaret",\
            "nautilus","octopus","okapi","pagoda","panda","pigeon","pizza",\
            "platypus","pyramid","revolver","rhino","rooster","saxophone",\
            "schooner","scissors","scorpion","sea_horse","snoopy","soccer_ball",\
            "stapler","starfish","stegosaurus","stop_sign","strawberry",\
            "sunflower","tick","trilobite","umbrella","watch","water_lilly",\
            "wheelchair","wild_cat","windsor_chair","wrench","yin_yang"]
        if not no_classes is None:
            classes = classes[-no_classes:]
            log.info(classes)
        self.class_order = classes
        train_set, test_set = self.select_data(classes,image_path,trainsize, \
            testsize)
        
        super(CaltechClassification,self).__init__(classes, train_set, test_set)
        

    def get_segmentation(self, im_path):
        """Implement this to return a segmentation (see segmentation.py) which
        is used in the construction of the estimators and in the testing.

        Keyword arguments:
        im_path -- the path to an image
        
        Returns:
        Segmentation object for the image.

        """
        return Classification(1)

    def get_ground_truth(self, im_path, segmentation):
        """Get the ground truth class for each segment in the segmentation. This
        is used in the construction of the estimators.

        Keyword arguments:
        im_path -- path to an image
        segmentation -- a Segmentation object for this image

        Returns:
        A list of class_names corresponding to the objects in the segmentation.
      
        """
        return [im_path.split('/')[-2]]
        
    def select_data(self, classes,image_path, trainsize, testsize):
        # Select files for each of 3 folders with images, of the sizes indicated
        train_set = []
        test_set = []
        files_per_class = trainsize + testsize
        for cls in classes:
            if not (cls == 'BACKGROUND_Google'):
                # Get all files in the path
                cl_files = os.listdir(image_path+'/'+cls)
                cl_paths = \
                    ['/'.join([image_path,cls,cl_file]) for cl_file in cl_files]
                # filter out all files to keep the jpg's only
                def filt(x): return re.search('\.jpg',x)
                cl_paths = filter(filt,cl_paths)
                # Shuffling the list first, and then take the first for the training set
                rndm.shuffle(cl_paths)
                train_set.extend(cl_paths[:trainsize])
                test_files = cl_paths[trainsize:files_per_class]
                test_set.extend(test_files)
                # Check whether the remainder of the files is enough for the test set
                if len(test_files) < testsize:
                    log.info("padding the testfiles of class %s"%cls +\
                        ": too little files: %d <%d"%(len(test_files),testsize))
                    double_files = test_files[:testsize - len(test_files)]
                    test_set.extend(double_files)
        # Return train and test set
        return train_set, test_set

class VOCResultsHandler(object):
    
    def __init__(self, dataset,results_path,th=0.5):
        self.classes = dataset.class_order
        self.results = dict()
        self.object_cls = dict()
        self.dataset = dataset
        self.results_path = results_path
        self.threshold = th
        for cls in self.classes:
            self.results[cls] = dict()
        os.mkdir('/'.join(self.results_path.split('/')[:-1]))

class VOCDetectionResultsHandler(VOCResultsHandler):
    
    def set_results(self,im_path,segmentation,results):
        log.debug('Segmentation: %s'%segmentation.bboxes)
        log.debug('  results: %s'%results)
        im_id = self.dataset.get_im_id(im_path)
        self.object_cls[im_id] = dict()
        for (bbox,result) in zip(segmentation.bboxes,results):
            if not result is None:
                mindist = float("inf")
                minclass = ''
                for (cls,dist) in result:
                    if not im_id in self.results[cls]:
                        self.results[cls][im_id] = []
                    self.results[cls][im_id].append((dist,bbox))
                    if dist < mindist:
                        mindist = dist
                        minclass = cls
                bb_string = str(bbox)
                self.object_cls[im_id][bb_string] = (minclass,mindist)
    
    def __str__(self):
        s = ''
        for (cls,imgs) in self.results.items():
            s += cls+':'
            for (img,bboxlist) in imgs.items():
                for (dist,bbox) in bboxlist:
                    s += "  %s %s %s %s %s %s"%( img, dist, bbox[0], bbox[1], \
                        bbox[2], bbox[3] )
    
    def save_to_files(self):
        mx = 0
        for (cls,imgs) in self.results.items():
            for (img,bboxlist) in imgs.items():
                for (dist,bbox) in bboxlist:
                    if not math.isinf(dist) and dist > mx:
                        mx = dist
        
        log.debug( 'Max: %f,thresh: %f,product: %f'%\
            (mx, self.threshold, mx*self.threshold) )
        for (cls,imgs) in self.results.items():
            cls_str = ""
            for (img,bboxlist) in imgs.items():
                for (dist,bbox) in bboxlist:
                    # Make a confidence value between 0-1
                    # Determined by the dist: dist/mx
                    log.debug( 'D: %f,norm: %f,conf: %f, maxed: %f'%( dist, \
                        (dist/(mx*self.threshold)), 1-(dist/(mx* self.threshold\
                        )), max(1-(dist/(mx*self.threshold)), 0)))
                    if not math.isinf(dist):
                        conf = max(1-(dist/(mx*self.threshold)),0)
                        if conf > 0:
                            cls_str += "%s %.5f %d %d %d %d\n"%(img, \
                            1 - (dist/mx), bbox[0], bbox[1], bbox[2], bbox[3])
            with open(self.results_path%cls ,'w') as f:
                f.write(cls_str)
        # Write a file for a confusion matrix: per file the class:
        object_cls_str = ""
        for im_id,bboxdict in self.object_cls.items():
            for bbox, clstuple in bboxdict.items():
                object_cls_str += "%s %s %s %f\n"%(im_id,bbox,clstuple[0],clstuple[1])
        with open('/'.join(self.results_path.split('/')[:-1])+'/objectclassification.txt','w') as f:
            f.write(object_cls_str)
        

class VOCClassificationResultsHandler(VOCResultsHandler):
    
    def set_results(self,im_path,_segmentation,results):
        log.info('Results to be set: %s'%results)
        im_id = self.dataset.get_im_id(im_path)
        mindist = float("inf")
        minclass = ''
        for (cls,dist) in results[0]:
            self.results[cls][im_id] = dist
            if dist < mindist:
                mindist = dist
                minclass = cls
        self.object_cls[im_id] = (minclass,mindist)
            
    
    def __str__(self):
        s = ''
        for (cls,imgs) in self.results.items():
            s += cls+':\n'
            for (img,dist) in imgs.items():
                s += "  %s %s\n"%( img, dist )
        
    def save_to_files(self):
        mx = 0
        for (cls,imgs) in self.results.items():
            for (img,dist) in imgs.items():
                if not math.isinf(dist) and dist > mx:
                    mx = dist
        
        log.debug( 'Max: %f,thresh: %f,product: %f'%\
            (mx, self.threshold, mx*self.threshold) )
        for (cls,imgs) in self.results.items():
            cls_str = ""
            for (img,dist) in imgs.items():
                # Make a confidence value between 0-1
                # Determined by the dist: dist/mx
                log.debug( 'D: %f,norm: %f,conf: %f, maxed: %f'%( dist, \
                    (dist/(mx*self.threshold)), 1-(dist/(mx* self.threshold\
                    )), max(1-(dist/(mx*self.threshold)), 0)))
                if not math.isinf(dist):
                    conf = max(1-(dist/(mx*self.threshold)),0)
                    if conf > 0:
                        cls_str += "%s %.5f\n"%( img, 1 - (dist/mx) )
            with open(self.results_path%cls ,'w') as f:
                f.write(cls_str)
        # Write a file for a confusion matrix: per file the class:
        object_cls_str = ""
        for im_id,clstuple in self.object_cls.items():
            object_cls_str += "%s %s %f\n"%(im_id,clstuple[0],clstuple[1])
        with open('/'.join(self.results_path.split('/')[:-1])+'/objectclassification.txt','w') as f:
            f.write(object_cls_str)
        

class CaltechResultsHandler(object):
    
    def __init__(self,dataset,results_path):
        self.results_path = results_path
        self.classes = dataset.class_order
        self.dataset = dataset
        self.results = dict()
        for cls1 in self.classes:
            self.results[cls1] = dict()
            for cls2 in self.classes:
                self.results[cls1][cls2] = []
        os.mkdir('/'.join(self.results_path.split('/')[:-1]))
    
    def set_results(self,im_path,segmentation,mxcls):
        gtcls = im_path.split('/')[-2]
        log.info('Results to be set: gt:%s, res:%s'%(gtcls,mxcls))
        self.results[gtcls][mxcls[0]].append(im_path)
        
    def __str__(self):
        s = ""
        for cls1 in self.classes:
            ss = ""
            for cls2 in self.classes:
                ss += "%d "% len(self.results[cls1][cls2])
            s += "%s\n"%ss
        return s
    
    def save_to_files(self):
        confmat = str(self)
        with open(self.results_path%'confmat','w') as f:
            f.write(confmat)

if __name__ == "__main__":
    
    import logging
    
    logging.basicConfig(level=logging.DEBUG)
    logging.info('Running Unit-test')
    classes = ['bird','train','bicycle','bottle','diningtable','person']
    trainsetfile = 'unittest_trsf.txt'
    testsetfile = 'unittest_tesf.txt'
    image_path = '../im/pascal11/VOCdevkit/VOC2011/JPEGImages/%s.jpg'
    gt_annotation_path = '../im/pascal11/VOCdevkit/VOC2011/Annotations/%s.xml'
    resultspath = 'tempresults/comp3_det_val_%s.txt'
    dataset = VOCDetection(classes,trainsetfile,testsetfile,image_path, \
        gt_annotation_path)
    logging.debug(dataset.train_set)
    descriptor = DescriptorUint8(cache_dir='.')
    estimator = NBNNEstimator.from_dataset('tempnbnn',dataset,descriptor)
    logging.info("======================STARTING TEST======================\n")
    
    vrh = VOCDetectionResultsHandler(dataset,resultspath,th=1)
    run_test(dataset, descriptor, estimator,vrh.set_results, \
        output_function=ranked_classify)
    print vrh
    vrh.save_to_files()
    gts = [dataset.get_ground_truth(ip,None) for ip in dataset.test_set]
    print gts 