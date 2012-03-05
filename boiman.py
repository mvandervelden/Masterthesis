# encoding: utf-8
"""
boiman.py

Created by Maarten van der Velden on 2012-02-21.
Copyright (c) 2012. All rights reserved.
"""

import os
import re
from numpy import *
import DescriptorIO, pyflann
from performance import *

class FFile(object):
    name = ''
    def __init__(self,name,path):
        self.name = name
        self.path = path
        self.base = name[:-4]
        self.ext = name[-4:]
    
    def __str__(self):
        return(self.path+self.name)
    
    def copy(self, fname=name):
        return FFile(fname, self.path)

class Test(object):
    class Params:
        pass
    
    def __init__(self, args):
        from ConfigParser import RawConfigParser
        
        config = RawConfigParser()
        self.config = config
        config.read(args.configfile)
        
        self.ID = args.ID
        self.test = config.get('Data', 'test')
        self.verbose = config.getboolean('General', 'verbose')
        
        if self.verbose:
            print 'Initializing Test'
        
        if args.descriptoronly:
            self.descr_only = True
            self.run_descriptors = True
            self.run_flann = False
            self.keep_descriptors = True
            self.trainsize = 0
            self.testsize = 0
            self.difficulty = ''
            print 'HOI!', self.keep_descriptors
        else:
            self.descr_only = False
            self.run_descriptors = True
            self.run_flann = True
            self.keep_descriptors = False
            self.trainsize = config.getint('Data','trainsize')
            self.testsize = config.getint('Data', 'testsize')
            self.difficulty = config.get('Data', 'difficulty')
            self.file_limits = {}
        
        r = config.get('General', 'resultsfile').split('.')
        self.resultsfile = r[0]+'_'+self.ID+'.'+r[1]
        self.tmp_dir = 'tmp_'+args.ID+'/'
                
        self.descr = self.Params()
        self.descr.descriptor = config.get('Descriptors', 'descriptor')
        self.descr.detector = config.get('Descriptors', 'detector')
        self.descr.binary = config.getboolean('Descriptors', 'save_binary')
        self.descr.scale_levels = config.getint('Descriptors', 'scale_levels')
        
        self.flann = self.Params()
        self.flann.k = config.getint('Flann', 'k')
    
    def select_files(self, path, sz):
        if self.verbose:
            print 'selecting {0} files from {1}'.format(sz,path)
        
        files = os.listdir(path)
        def filt(x): return re.search('.jpg',x)
        files = filter(filt,files)
        
        if not self.descr_only:
            limit = [v for k,v in self.file_limits.items() if k in path]
            if not limit == []:
                limit = limit[0]
                def rm_limit(x, limit): return int(re.search('[0-9]+',x).group(0)) <= limit
                files = [f for f in files if rm_limit(f, limit)]
                
            random.shuffle(files)
            files = files[:sz]
        
        files = [FFile(f, path) for f in files]
        return files
    
    def get_all_descriptors(self):
        if self.verbose:
            print 'Getting Descriptors'
        
        from subprocess import Popen, PIPE
        
        if not self.tmp_dir[:-1] in os.listdir('.'):
            os.mkdir(self.tmp_dir)
        
        self.outputbase = self.descr.detector + "_" + self.descr.descriptor + "_"
        if self.descr.binary:
            self.dopts = [  "--detector", self.descr.detector, "--descriptor", \
                            self.descr.descriptor, "--outputFormat", 'binary',"--output"]
            self.outputext = '.dbin'
        else:
            self.dopts = [  "--detector", self.descr.detector, "--descriptor", \
                            self.descr.descriptor,"--output"]
            self.outputext = '.dtxt'
        
        if self.descr_only:
            if self.verbose: print 'getting all descriptors: running descriptor only'
            self.get_descriptor_sets(self.fileset)
        else:
            if self.verbose: print 'getting training descriptors:'
            self.train_descriptors = [array(0)]*self.train_set.shape[0]
            for it, clfiles in enumerate(self.train_set):
                dscr, _ = self.get_descriptor_sets(clfiles)
                self.train_descriptors[it] = dscr
            if self.verbose: print 'getting test descriptors:'
            self.test_descriptors, self.dssize = self.get_descriptor_sets(self.test_set)        

    def get_descriptor_sets(self,fileset, ):
        if self.descr.scale_levels > 1:
            import cv2.cv as cv
        
        no_files = len(fileset)
            
        if self.run_flann:
            descriptors = [array(0) for i in range(no_files*self.descr.scale_levels)]
        if self.verbose: print '....file:', 
        for i, f in enumerate(fileset):
            if self.verbose: print i,
            if self.keep_descriptors:
                tmp_dir = f.path + 'descriptors/'
                if not 'descriptors' in os.listdir(f.path):
                    os.mkdir('descriptors')
            else:
                tmp_dir = self.tmp_dir
            
            if self.run_flann:
                descriptors[i*self.descr.scale_levels] = self.run_color_descr(f, tmp_dir)
            else:
                self.run_color_descr(f, tmp_dir)

            if self.descr.scale_levels > 1:
                im1 = cv.LoadImageM(str(f))
                base = f.base
                for scale in range(1,self.descr.scale_levels):
                    im2 = cv.CreateMat(im1.rows/2,im1.cols/2,im1.type)
                    cv.PyrDown(im1, im2)
                    f = FFile(base+'_'+str(scale)+f.ext, self.tmp_dir)
                    cv.SaveImage(str(f),im2)
                    
                    if self.run_flann:
                        descriptors[i*self.descr.scale_levels+scale] = self.run_color_descr(f,tmp_dir)
                    else:
                        self.run_color_descr(f, tmp_dir)
                    os.remove(str(f))
                    im1 = im2
        if self.verbose: print '' 
        if self.run_flann:
            return vstack(descriptors), array([d.shape[0] for d in descriptors]).reshape(self.descr.scale_levels,no_files).sum(0)
            
    def run_color_descr(self, f, tmp_dir):
        from subprocess import Popen, PIPE
        
        
        o = tmp_dir+self.outputbase+f.base+self.outputext
        print o
        run_args= ['colorDescriptor', str(f)] + self.dopts + [o]
        #print run_args
        #print type(run_args)
        p = Popen(run_args,stdout=PIPE, stderr=PIPE)
        _,err = p.communicate()
        if not err == '':
            raise Exception("ColDescriptor run failed.\n Message: {0}\n Did not make output for {1}".format(err, f))
        else:
            if not self.descr_only:
                # read descriptors:
                _, d = DescriptorIO.readDescriptors(o)
                if not self.keep_descriptors:
                    os.remove(o)
                return d
            else:
                if not self.keep_descriptors:
                    os.remove(o)
    
    def find_nn(self):
        if self.verbose: print 'finding nearest neighbors.'
        self.distances = zeros([self.no_classes,self.test_descriptors.shape[0]])
        for it, descr in enumerate(self.train_descriptors):
            if self.verbose: print '...class: ', it
            flann = pyflann.FLANN()
            print self.distances.shape
            _, d = flann.nn(descr, self.test_descriptors, \
                        num_neighbors=self.flann.k, algorithm='kdtree')
            print d.shape
            self.distances[it,:] = d
        
    def get_class(self):
        if self.verbose: print 'Finding NB class.'
        no_files = self.classification.shape[0]
        # nns is a n x m array, where n=number of classes, m=numbr of test descriptors
        # dssize = no of features per image
        # testsz = no of test images
        starts = hstack([0, cumsum(self.dssize)])
        #print dssize, starts,nns.shape
        self.c_hat = zeros(no_files,int)
        
        if self.verbose: print 'file: ',
        for i in xrange(no_files):
            if self.verbose: print i,
            nfile = self.distances[:,starts[i]:starts[i+1]]
            nsum = sum(nfile**2,1)
            self.c_hat[i] = argmin(nsum)
        if self.verbose: print ''
        return self.c_hat
    
    def save_results(self):
        print 'Saving results to: ', self.resultsfile
        self.config.add_section('Results')
        self.config.set('Results','ground_truth', self.classification)
        self.config.set('Results', 'classification', self.c_hat)
        
        with open(self.resultsfile,'wb') as cfgfile:
            self.config.write(cfgfile)
    
class GrazTest(Test):
    
    def __init__(self, args):
        super(GrazTest, self).__init__(args)
        self.no_classes = 2
        if self.verbose: print 'Test will be GrazTest'
        if self.difficulty == 'no_hard':
            self.file_limits = {'persons':348, 'bikes': 306, 'no_bike_no_person': 273}
    
    def select_data(self, pathp, pathn1, pathn2):
        if self.verbose: print 'Selecting Data the Graz01 way..'
        
        pos_files = array(self.select_files(pathp,  self.trainsize + self.testsize))
        n1_files= array(self.select_files(pathn1, (self.trainsize + self.testsize)/2))
        n2_files  = array(self.select_files(pathn2, (self.trainsize + self.testsize)/2))
        
        if not self.descr_only:
            self.train_set = vstack([pos_files[:self.trainsize], hstack([n1_files[:self.trainsize/2],n2_files[:self.trainsize/2]])])
            self.test_set  = hstack([pos_files[self.trainsize:],n1_files[self.trainsize/2:], n2_files[self.trainsize/2:]])
            self.classification = array([0]*self.testsize + [1]*self.testsize, int)
            return self.classification
        else:
            self.fileset = hstack([pos_files,n1_files, n2_files])
        
class GrazPersonTest(GrazTest):
    
    def select_data(self):
        if self.verbose: print 'Selecting data the GrazPersonTest way'
        
        pathp = "../im/graz01/persons/"
        pathn1 = "../im/graz01/bikes/"
        pathn2 = "../im/graz01/no_bike_no_person/"
        return super(GrazPersonTest, self).select_data(pathp,pathn1,pathn2)
        

class GrazBikeTest(GrazTest):
    
    def select_data(self):
        if self.verbose: print 'Selecting data the GrazBikeTest way'
        
        pathp = "../im/graz01/bikes/"
        pathn1 = "../im/graz01/persons/"
        pathn2 = "../im/graz01/no_bike_no_person/"
        return super(GrazBikeTest, self).select_data(pathp,pathn1,pathn2)


class CaltechTest(Test):
    pass

class PascalTest(Test):
    pass

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--ID', default='1')
    parser.add_argument('TEST')
    parser.add_argument('-c', '--configfile', default="settings.cfg")
    parser.add_argument('-s', '--descriptoronly', action='store_true')
    args = parser.parse_args()
    if args.TEST == 'graz01_person':
        test = GrazPersonTest(args)
    elif args.TEST == 'graz01_bike':
        test = GrazBikesTest(args)
    elif args.TEST == 'caltech101':
        test = CaltechTest(args)
    elif args.TEST == 'pascal':
        test = PascalTest(args)
    elif args.TEST == 'graz01_descriptor':
        test = GrazTest(args)

    c = test.select_data()
    
    c_hats = zeros(test.no_classes*test.testsize)
    classes = zeros(test.no_classes*test.testsize)
        
    if test.run_descriptors:
        test.get_all_descriptors()
    if test.run_flann:
        test.find_nn()
        c_hats = test.get_class()
        test.save_results()
        cfmat = get_confmat(classes, c_hats, test.no_classes)
        print cfmat
        roc = get_ROC_equal_rate(cfmat)
        print 'ROC: ', roc
    