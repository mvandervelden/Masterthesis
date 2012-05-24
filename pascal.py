from nbnn import *
import xml.parsers.expat as expat
import re,os,math

log = logging.getLogger(__name__)

class VOCDetection(VOC):
    
    def __init__(self, classes, trainset_file, testset_file, image_path,
            gt_annotation_path):
        log.info("Creating VOCDetection.")
        super(VOCDetection,self).__init__(classes, trainset_file, \
                testset_file, image_path)
        self.gt_annotation_path = gt_annotation_path
    
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
                
                self.blist = ['xmin','ymin','xmax','ymax']
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

                self.depth += 1
                log.debug('start_element %d: %s'%(self.depth, name))
                
                self.cur_field = name
                if self.depth == 1 and name == 'object':
                    self.cur_object = dict()
                elif self.depth == 2 and name == 'bndbox':
                    self.cur_object['bbox'] = [0]*4
                
            def end_element(self,name):
                """ Calback function when the expat parser finds the end of an
                element. Keeps track of how deep we are in the xml structure,
                and stores a completely object in the object list if needed.
                
                Keyword arguments:
                name - the name of the element that is ended.
                
                """

                # logging.debug('end_element: %d: %s'%(self.depth,name))

                self.depth -= 1
                if self.depth == 0 and name == 'object':
                    log.debug(self.cur_object)
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
                data = data.strip()
                # print 'cd',repr(data),'dpth',self.depth
                if not self.cur_object is None and not data == '':
                    if self.depth == 2:
                        if self.cur_field == 'name':
                            log.debug('char_data (name): %s'%(data))
                            self.cur_object[self.cur_field] = data
                        elif self.cur_field == 'difficult':
                            log.debug('char_data (diff): %s'%(data))
                            self.cur_object[self.cur_field] = data == '1'
                            # else:
                            #     self.cur_object[self.cur_field] = data
                    elif self.depth == 3 and \
                        self.cur_field in ['xmin','ymin','xmax','ymax']:
                        log.debug('char_data (bbox): %s'%(data))
                        idx = self.blist.index(self.cur_field)
                        self.cur_object['bbox'][idx] = int(data)
            
            def get_objects(self):
                """Getter that returns the list of objects.
                
                """
                
                return self.objects

        p = expat.ParserCreate()
        xp = XParser()
        p.StartElementHandler = xp.start_element
        p.EndElementHandler = xp.end_element
        p.CharacterDataHandler = xp.char_data
        with open(annotation_file,'r') as af:
            p.ParseFile(af)
        
        return xp.get_objects()

class VOCResultsHandler(object):
    
    def __init__(self, dataset,results_path,th=0.5):
        self.classes = dataset.class_order
        self.results = dict()
        self.dataset = dataset
        self.results_path = results_path
        self.threshold = th
        for cls in self.classes:
            self.results[cls] = dict()
        os.mkdir('/'.join(self.results_path.split('/')[:-1]))
        
    def voc_detection_results(self,im_path,segmentation,results):
        im_id = self.dataset.get_im_id(im_path)
        for (bbox,result) in zip(segmentation.bboxes,results):
            for (cls,dist) in result:
                if not im_id in self.results[cls]:
                    self.results[cls][im_id] = []
                self.results[cls][im_id].append((dist,bbox))
    
    def print_results(self):
        for (cls,imgs) in self.results.items():
            print cls+':'
            for (img,bboxlist) in imgs.items():
                for (dist,bbox) in bboxlist:
                    print "  %s %s %s %s %s %s"%( img, dist, bbox[0], bbox[1], \
                        bbox[2], bbox[3] )
    
    def save_results(self):
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
    
    vrh = VOCResultsHandler(dataset,resultspath,th=1)
    run_test(dataset, descriptor, estimator,vrh.voc_detection_results, \
        output_function=ranked_classify)
    vrh.print_results()
    vrh.save_results()
    gts = [dataset.get_ground_truth(ip,None) for ip in dataset.test_set]
    print gts 