from nbnn import *
from numpy import random as rndm

class GrazTest(Test):
    
    def __init__(self, output_dir, descriptors, trainsize, testsize, 
            filetype='png',test='person',difficult=False, flann_args={}):
        self.trainsize = trainsize
        self.testsize = testsize
        self.filetype = filetype
        self.test = test
        if not difficult:
            self.file_limits = {'persons':348, 'bikes': 306, \
                'no_bike_no_person': 273}
        else:
            self.file_limits = {}
        
        super(GrazTest,self).__init__(output_dir, descriptors, flann_args)
    
    def select_data(self):
        
        if os.path.exists('scratchdisk/im/graz01'+self.filetype):
            basepath = 'scratchdisk/im/graz01'+self.filetype
        else:
            basepath = '../im/graz01'+self.filetype
        # On the nodes, the images should be on the scratch disk
        if self.test == 'person':
            pathp = basepath+'/persons'
            pathn1 = basepath+'/bikes'
        elif self.test == 'bike':
            pathp = basepath+'/bikes'
            pathn1 = basepath+'/persons'
        pathn2 = basepath+'/no_bike_no_person'
        
        trp, tep = self.get_files(pathp, self.trainsize, self.testsize)
        trn1, ten1 = self.get_files(pathn1, self.trainsize/2, self.testsize/2)
        trn2, ten2 = self.get_files(pathn2, self.trainsize/2, self.testsize/2)
        
        return trp+trn1+trn2, tep+ten1+ten2
        
    def get_files(self, path, s1, s2):
        import re
        
        files = os.listdir(path)
        paths = [path+'/'+f for f in files]
        print self.filetype
        def filt(x): return re.search('\.'+self.filetype,x)
        paths = filter(filt,paths)
        
        limit = [v for k,v in self.file_limits.items() if k in path]

        if not limit == []:
            # If there's a limit, filter out the images exceeding this limit
            limit = limit[0]
            def rm_limit(x, limit): return int(re.search('[0-9]+',x).group(0)) <= limit
            paths = [p for p in paths if rm_limit(p.split('/')[-1], limit)]
        
        rndm.shuffle(paths)
        return paths[:s1], paths[s1:s1+s2]
    
    def get_ground_truth(self, im_path):
        # Set the ground truth of an image: the class is the name of the folder in which the image is.
        if self.test == 'person':
            classes = ['persons', 'nonpersons']
        elif self.test == 'bike':
            classes = ['bikes', 'nonbikes']
        folder = im_path.split('/')[-2]
        if folder == classes[0]:
            cls = classes[0]
        else:
            cls = classes[1]
        return {cls: [ImageObject(im_path)]}

if __name__ == "__main__":
    
    dargs = [{"cache_dir": "./", 'verbose':True}]
    descriptors = [Descriptor(**kwargs) for kwargs in dargs]
    test = GrazTest('./test',descriptors, 2,2, test='bike',difficult=True, \
        flann_args={"verbose":True})
    result = test.run_test(nbnn_classify)
    print [test.get_ground_truth(imp).keys() for imp in test.test_set]
    print result
    