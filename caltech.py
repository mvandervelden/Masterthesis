from nbnn import *
from numpy import random as rndm
class CaltechTest(Test):
    
    def __init__(self, output_dir, descriptors, trainsize, testsize, 
            no_classes=101,flann_args={}):
        self.trainsize = trainsize
        self.testsize = testsize
        self.no_classes = no_classes
        super(CaltechTest,self).__init__(output_dir, descriptors, flann_args)
    
    def select_data(self):
        import re
        
        if os.path.exists('scratchdisk/im/caltech101/101_ObjectCategories'):
            # On the nodes, the images should be on the scratch disk
            motherpath = 'scratchdisk/im/caltech101/101_ObjectCategories'
        else:
            motherpath = '../im/caltech101/101_ObjectCategories'
        classlist = os.listdir(motherpath)
        if self.no_classes < 101:
            classlist = classlist[:self.no_classes]
        # Select files for each of 3 folders with images, of the sizes indicated
        train_set = []
        test_set = []
        files_per_class = self.trainsize+self.testsize
        for cls in classlist:
            if not (cls == 'BACKGROUND_Google'):
                # Get all files in the path
                cl_files = os.listdir(motherpath+'/'+cls)
                cl_paths = \
                    ['/'.join([motherpath,cls,cl_file]) for cl_file in cl_files]
                # filter out all files to keep the jpg's only
                def filt(x): return re.search('\.jpg',x)
                cl_paths = filter(filt,cl_paths)
                # Shuffling the list first, and then take the first for the training set
                rndm.shuffle(cl_paths)
                train_set.extend(cl_paths[:self.trainsize])
                test_files = cl_paths[self.trainsize:files_per_class]
                test_set.extend(test_files)
                # Check whether the remainder of the files is enough for the test set
                if len(test_files) < self.testsize:
                    print "padding the testfiles of class "+cls+\
                        ": too little files:", len(test_files), "<", \
                        self.testsize
                    double_files = test_files[:self.testsize - len(test_files)]
                    if len(double_files) < self.testsize - len(test_files):
                        raise Exception("More than once too little files")
                    test_set.extend(double_files)
        # Return train and test set
        return train_set, test_set
    
    def get_ground_truth(self, im_path):
        # Set the ground truth of an image: the class is the name of the folder in which the image is.
        cls = im_path.split('/')[-2]
        return {cls: [ImageObject(im_path)]}

if __name__ == "__main__":
    
    dargs = [{"cache_dir": "./"}]
    descriptors = [Descriptor(**kwargs) for kwargs in dargs]
    test = CaltechTest('./test',descriptors, 1,1,10)
    result = test.run_test(nbnn_classify)
    print [test.get_ground_truth(imp).keys() for imp in test.test_set]
    print result
    