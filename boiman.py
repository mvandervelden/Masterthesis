#! /usr/bin/env python
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
from subprocess import Popen, PIPE
import shutil

class FFile(object):
    """ Class for easily representing files, having their path, extension and extension available separately """
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
    """ Main Test class, having most of the functions, which are the same over all kinds of data sets """
    
    class Params:
        # Struct-like class to put arguments in
        pass
    
    def __init__(self, args, test):
        from ConfigParser import RawConfigParser
        
        self.test = test
        
        # Get the settings from the configfile specified (settings.cfg by default)
        if not isinstance(args.configfile, RawConfigParser):
            config = RawConfigParser()
            self.config = config
            config.read(args.configfile)
        else:
            config = args.configfile
        
        self.ID = args.ID
        self.verbose = config.getboolean('General', 'verbose')
        self.no_classes = args.no_classes
        if self.verbose:
            print 'Initializing Test'
        
        # is always True at start, wil be set to False afterwards, so the descriptors of the first test file will be saved
        self.keep_descriptors = False
        # Same goes for this, keep the nearest neighbors for the first test file
        self.keep_nn = True
        self.sample = []
        
        # Set all parameters
        # Size of the train and test sets samples, per class
        self.trainsize = config.getint('Data','trainsize')
        self.testsize = config.getint('Data', 'testsize')
        # Set any restrictions on the difficulty of the image that can be selected. These settigns are test-dependent, and the file_limits dict is used to set restrictions on the files that can be selected per class
        self.difficulty = config.get('Data', 'difficulty')
        self.file_limits = {}
        
        # Try to set the temporary folder on the RAMdisk (/dev/shm/), only available on Linux. If not available, put it in a subfolder tmp_$ID
        if os.path.exists('/dev/shm'):
            if self.verbose: print 'Using RAMdisk'
            self.tmp_dir = '/dev/shm/tmp_'+args.ID+'/'
        else:
            self.tmp_dir = 'tmp_'+args.ID+'/'
        # make sure the temporary folder exists and is empty
        if os.path.exists(self.tmp_dir):
            ls = os.listdir(self.tmp_dir)
            if len(os.listdir(self.tmp_dir)) > 0:
                for f in ls:
                    os.remove(self.tmp_dir+f)
        else:
            os.mkdir(self.tmp_dir)
        # Set up a results folder
        if os.path.exists('/var/scratch/vdvelden'):
            self.resultsdir = '/var/scratch/vdvelden/' + self.test + '_results/'
            if not self.test+'_results' in os.listdir('/var/scratch/vdvelden'):
                os.mkdir(self.resultsdir)
        else:
            self.resultsdir = self.test+'_results/'
            if not self.resultsdir[:-1] in os.listdir('.'):
                os.mkdir(self.resultsdir)
        # Make a file name for the general results
        r = config.get('General', 'resultsfile').split('.')
        self.resultsfile =self.resultsdir + r[0]+'_'+self.ID+'.'+r[1]
        
        # Set all descriptor parameters in a Params struct
        self.descr = self.Params()
        self.descr.descriptor = config.get('Descriptors', 'descriptor')
        self.descr.detector = config.get('Descriptors', 'detector')
        self.descr.binary = config.getboolean('Descriptors', 'save_binary')
        self.descr.scale_levels = config.getint('Descriptors', 'scale_levels')
        self.descr.scale_base = config.getfloat('Descriptors', 'scale_base')
        self.descr.spacing_base = config.getint('Descriptors', 'spacing_base')
        
        # Set all FLANN parameters in a Params struct
        self.flann = self.Params()
        self.flann.k = config.getint('Flann', 'k')
        
        
    
    def select_files(self, path, sz):
        """ Function to select 'sz' files from a path, at random"""
        if self.verbose:
            print 'selecting {0} files from {1}'.format(sz,path)
        
        # Get all files in the path
        files = os.listdir(path)
        # filter out all files to keep the jpg's only
        def filt(x): return re.search('.jpg',x)
        files = filter(filt,files)
        
        # look up whether there are limits on the files that can be used in this class (there should be at most 1 key in the file_limits dict that matches the current path)
        # Usually this is used to filter out hard to classify images
        limit = [v for k,v in self.file_limits.items() if k in path]
        if not limit == []:
            # If there's a limit, filter out the images exceeding this limit
            limit = limit[0]
            def rm_limit(x, limit): return int(re.search('[0-9]+',x).group(0)) <= limit
            files = [f for f in files if rm_limit(f, limit)]
        
        # Select sz files at random by shuffling the list first, and then taking the first sz
        if sz <= len(files):
            random.shuffle(files)
            files = files[:sz]
        else:
            # The number of files to select is less than the required size:
            random.shuffle(files)
            plusfiles = files[:sz-len(files)]
            files += plusfiles
            random.shuffle(files)
            print 'too little files, padding'
        
        # make FFile instances for all files selected
        files = [FFile(f, path) for f in files]
        return files
    
    def set_descriptor_variables(self):
        """ Function to set the options and output variables that are the same for training files, test files and classes"""
        if self.verbose:
            print 'Setting Descriptor variables'
        
        #Set the base for the filename of the descriptors that will be calculated
        self.outputbase = self.descr.detector + "_" + self.descr.descriptor + "_"
        # Set the options for colorDescriptor in a list, to be passed to Popen
        # Also, set the extension of the descriptor file that will be used
        # These settings depend on the fact whether the descriptors will be saved binary or in text
        if self.descr.binary:
            self.dopts = [  "--detector", self.descr.detector, "--descriptor", \
                            self.descr.descriptor, "--outputFormat", 'binary']
            self.outputext = '.dbin'
        else:
            self.dopts = [  "--detector", self.descr.detector, "--descriptor", \
                            self.descr.descriptor]
            self.outputext = '.dtxt'
        
    def get_descriptors(self, fileset, save_dssize=False):
        """ Function to create descriptors for a number of files in a list of files fileset, 
            the number of descriptors per image can be saved to self.dssize, when the flag save_dssize is set.
            The function returns an array of descriptors"""
        if self.verbose: print 'Descriptors:',
        
        no_files = len(fileset)
        #Initialize a list of empty arrays, on entry for each scale of each file
        descriptors = [array(0,uint8) for i in range(no_files*self.descr.scale_levels)]
        if self.verbose: print '....file:' 
        
        # Iterate over the files in the set
        siz = 0
        for i, f in enumerate(fileset):
            if self.verbose: print i
            # If this is the first file, make sure to keep the discriptors and set the flag to False
            if self.keep_descriptors:
                save=True
            else:
                save=False
            # Iterate over the scales at which descriptors have to be calculated
            for scale in range(self.descr.scale_levels):
                # Determine the scaling factor ({base, base^2, ... base^no_scales})
                scaling = self.descr.scale_base**(scale+1)
                # Determine the spacing factor ({base, base*scalebase, base*scalebase^2, ..., base * scalebase ^(no_scale-1)}, rounded to the lower amount of pixels: "better to have a bit more descriptors than a bit less"..)
                spacing = int(floor(self.descr.spacing_base*(self.descr.scale_base**scale)))
                # Calculate the descriptors with the given parameters and store the resulting array in the descriptors-list
                dd = self.run_color_descr(f, self.tmp_dir, scaling, spacing, save)
                siz += (dd.shape[0]*dd.shape[1]*8)/8000000
                descriptors[(i*self.descr.scale_levels)+scale] = dd
            print siz,'MB'
        # Save the number of descriptors of each file into self.dssize if the save_dssize flag is set
        if save_dssize:
            # get the size of each array in the list of descriptors, and sum the amounts for each file (over all scale levels, that's wehre the reshape is for)
            self.dssize = array([d.shape[0] for d in descriptors]).reshape(self.descr.scale_levels,no_files).sum(0).astype(uint32)
        # Convert the list of descriptors to a 2D array and return this, which file or scale it is doesn't matter anymore.
        return vstack(descriptors)
    
    def run_color_descr(self, f, tmp_dir, scaling, spacing, save):
        """ Function that calls the command colorDescriptor for a file f, using a certain scaling and spacing, storing the result
            temporarily into tmp_dir, and saving the descriptor files to the results directory when 'save' is set.
            The resulting descriptors are loaded and returned."""
        
        # Build the filename of the (temporary) descriptor file
        o = self.outputbase+'sc{0:.1f}'.format(scaling)+'_sp'+str(spacing)+'_'+f.base+self.outputext
        p_o = tmp_dir + o
        # Make a list of the command and all parameters needed
        run_args= ['colorDescriptor', str(f)] + self.dopts + ['--ds_spacing', str(spacing), '--ds_scales', '{0:.1f}'.format(scaling), '--output', p_o]
        #print run_args
        #print type(run_args)
        
        # Call the colorDescriptor command with arguments, discard all output (goes to PIPE)
        p = Popen(run_args,stdout=PIPE, stderr=PIPE)
        # Check whether errors were encountered
        _,err = p.communicate()
        if not err == '':
            raise Exception("ColDescriptor run failed.\n Message: {0}\n Did not make output for {1}".format(err, f))
        else:
            # The descriptors are read again using Koen's DescriptorIO implementation
            # We only need the descriptors d, we discard the location and scale returned as the first argument
            _, d = DescriptorIO.readDescriptors(p_o)
            # If the save flag is on, the descriptor file will not be removed, but it will be saved to the results folder, and its new location will be added to self.sample, which will de added to the resultsfile
            if not save:
                os.remove(p_o)
            else:
                self.sample.append(self.resultsdir+o)
                shutil.move(p_o, self.resultsdir)
            # return the descriptors
            
            # Convert to 8bit uints, take care of possible 512 values
            d=dstack([d/2,ones(d.shape,uint8)*127]).min(2).astype(uint8)
            return d
    
    def find_nn(self, test_descr, train_descr):
        """ Call FLANN to calculate the nearest neighbors of test_descr to the train_descr, return the distances d to the nearest neighbor
            and the indexes of the features f. """
        if self.verbose: print 'finding nearest neighbors.'
        # Initialize FLANN 
        flann = pyflann.FLANN()
        # Get the nearest neighbors using the FLANN parameters (can also be used to make an index, to be used later on, but right now I just discard the index immediately after using it)
        f, d = flann.nn(train_descr, test_descr, num_neighbors=self.flann.k, algorithm='kdtree')
        f = f.astype(uint32)
        return f,d
        
    def get_class(self, nns):
        """ Function to get the most likely class in an array nns (n x m), where n = the number of classes, and m is the number
            of descriptors. It returns an i-sized vector c_hat, with the classification for each test image i"""
        if self.verbose: print 'Finding NB class.'
        
        no_files = self.classification.shape[0]
        # Define which descriptors (distance values) belong to which test image by referring to self.dssize.
        # The first descriptor (distance) of each file is defined in 'starts', which is length no_files+1 (the last entry equals m)
        starts = hstack([0, cumsum(self.dssize)])
        # Initialize the classification
        if self.no_classes < 256:
            dt = uint8
        else:
            dt = uint16
        self.c_hat = zeros(no_files, dt)
        
        if self.verbose: print 'file: ',
        # Iterate over the files, and for each file, get it's descriptors distances to all classes from nns[:,xx],
        # take the sum of the squares of these distances per class and then use the argmin as the NB assumption of the most likely class in c_hat
        print 'no_files: ', no_files, 'starts_len: ',starts.shape
        for i in xrange(no_files):
            if self.verbose: print i,
            print 'start-end:',starts[i],'-', starts[i+1], 'nnssize: ',nns.shape
            nfile = nns[:,starts[i]:starts[i+1]]
            nsum = sum(nfile**2,1)
            self.c_hat[i] = argmin(nsum)
        if self.verbose: print ''
        # Return the classification for each file
        return self.c_hat
    
    def save_results(self, train_set, test_set, nns):
        """ Save the results of the experiment, along with a list of files used, the distances found, and the indexes of the 
            features. It stores the experiment settings to a cPickle (.pkl) file with actual results."""
        print 'Saving results to: ', self.resultsfile
        
        
        import cPickle as cpk
        with open(self.resultsfile, 'wb') as cpkfile:
            # Make a cPickle file and dump all results into it
            cpk.dump(self.config, cpkfile)

            cpk.dump(self.classification, cpkfile)
            cpk.dump(self.c_hat, cpkfile)
            cpk.dump([[str(f) for f in ff] for ff in train_set], cpkfile)
            cpk.dump([str(f) for f in test_set],cpkfile)
            cpk.dump(nns,cpkfile)
            cpk.dump(self.dssize,cpkfile)
            cpk.dump(self.classlist,cpkfile)

    def remove_tmpfiles(self):
        # Clean up
        if self.verbose: print 'Removing Temp files'
        
        # If the tmp_dir is not empty yet, empty its contents, and then remove the tmp_dir itself
        if not os.listdir(self.tmp_dir) == []:
            os.remove(self.tmp_dir+'*')
        shutil.rmtree(self.tmp_dir)
    
    def predict_descriptor_sizes(self, no_test_files):
        spacing = zeros(self.descr.scale_levels)    
        for i,scale in enumerate(range(self.descr.scale_levels)):
            # Determine the spacing factor ({base, base*scalebase, base*scalebase^2, ..., base * scalebase ^(no_scale-1)}, rounded to the lower amount of pixels: "better to have a bit more descriptors than a bit less"..)
            spacing[i] = (int(floor(self.descr.spacing_base*(self.descr.scale_base**scale))))
        dim = self.avg_filesize
        img_no_descr = outer(dim,1./spacing).round().prod(0).sum()
        Mbytes_per_image = int(img_no_descr*128)/1000000
        total_Mbytes = no_test_files*Mbytes_per_image
        print total_Mbytes, 'M'
        if os.getlogin() == 'iMaarten':
            self.max_size = 2000
        elif os.getlogin() == 'vdvelden':
            self.max_size = 20000
        else:
            self.max_size = 2000
        files_per_chunk = self.max_size/Mbytes_per_image
        return files_per_chunk

        
    
class GrazTest(Test):
    """ Class with data-specific functions for the Graz01 data set"""
    def __init__(self, args, test):
        # Do a default intialization
        super(GrazTest, self).__init__(args, test)
        # Set the number of classes to 2 (positive, negative)
        if self.no_classes == 0 or self.no_classes > 2:
            self.no_classes = 2
        if self.verbose: print 'Test will be GrazTest'
        # This data set can be run with difficulty 'no_hard' of full. In the former case only the first n images per
        # image folder can be chosen, because, following the data manual, the ones with an index above this are hard to classify.
        if self.difficulty == 'no_hard':
            self.file_limits = {'persons':348, 'bikes': 306, 'no_bike_no_person': 273}
        
        self.avg_filesize = array([640,480])
        
    def select_data(self, pathp, pathn1, pathn2):
        """ Select data using the set-up used in the Boiman experiment with Graz01 data. 
            For the positive class, trainsize+testsize items are selected, and then subdivided into the respective parts
            of the train and test set. The negative class consist of files from two folders, equally divided."""
        if self.verbose: print 'Selecting Data the Graz01 way..'
        
        # Select files for each of 3 folders with images, of the sizes indicated
        pos_files = array(self.select_files(pathp,  self.trainsize + self.testsize))
        n1_files= array(self.select_files(pathn1, (self.trainsize + self.testsize)/2))
        n2_files  = array(self.select_files(pathn2, (self.trainsize + self.testsize)/2))
        
        # Divide the three arrays into a train_set and a test_set. The train_set is 2D (two classes), the test_set is 1D (their class doesn't matter) 
        train_set = vstack([pos_files[:self.trainsize], hstack([n1_files[:self.trainsize/2],n2_files[:self.trainsize/2]])])
        test_set  = hstack([pos_files[self.trainsize:],n1_files[self.trainsize/2:], n2_files[self.trainsize/2:]])
        # Set the ground truth of the classification of the test images
        self.classification = array([0]*self.testsize + [1]*self.testsize, uint8)
        # Return train and test set
        return train_set, test_set
        
class GrazPersonTest(GrazTest):
    """ Class with data_specific functions for a Graz01Person test, defined in Boiman"""
    def __init__(self,args):
        # Call the generic constructor, defining the test as a 'person' test
        super(GrazPersonTest, self).__init__(args,'graz01_person')
        self.classlist = ['person','no_person']
    
    def select_data(self):
        """ Select data, where the positive path is the one with person images, and the two negative paths have bikes, 
            and neither persons nor bikes, respectively."""
        if self.verbose: print 'Selecting data the GrazPersonTest way'
        
        if os.path.exists('/var/scratch/vdvelden/im/graz01'):
            # On the nodes, the images should be on the scratch disk
            pathp = "/var/scratch/vdvelden/im/graz01/persons/"
            pathn1 = "/var/scratch/vdvelden/im/graz01/bikes/"
            pathn2 = "/var/scratch/vdvelden/im/graz01/no_bike_no_person/"
        else:
            pathp = "../im/graz01/persons/"
            pathn1 = "../im/graz01/bikes/"
            pathn2 = "../im/graz01/no_bike_no_person/"
        return super(GrazPersonTest, self).select_data(pathp,pathn1,pathn2)
        

class GrazBikeTest(GrazTest):
    """ Class with data_specific functions for a Graz01Bike test, defined in Boiman"""
    def __init__(self,args):
        # Call the generic constructor, defining the test as a 'person' test
        super(GrazBikeTest, self).__init__(args,'graz01_bike')
        self.classlist = ['bike','no_bike']
    
    def select_data(self):
        """ Select data, where the positive path is the one with bike images, and the two negative paths have persons, 
            and neither persons nor bikes, respectively."""
        if self.verbose: print 'Selecting data the GrazBikeTest way'
        
        if os.path.exists('/var/scratch/vdvelden/im/graz01'):
            # On the nodes, the images should be on the scratch disk
            pathp = "/var/scratch/vdvelden/im/graz01/bikes/"
            pathn1 = "/var/scratch/vdvelden/im/graz01/persons/"
            pathn2 = "/var/scratch/vdvelden/im/graz01/no_bike_no_person/"
        else:
            pathp = "../im/graz01/bikes/"
            pathn1 = "../im/graz01/persons/"
            pathn2 = "../im/graz01/no_bike_no_person/"
            
        return super(GrazBikeTest, self).select_data(pathp,pathn1,pathn2)


class CaltechTest(Test):
    """ Class with data_specific functions for a Caltech101 test, defined in Boiman"""
    def __init__(self,args):
        super(CaltechTest, self).__init__(args,'caltech101')
        
        if self.no_classes == 0:
            if self.difficulty == 'no_background':
                #self.no_classes = 101
                self.no_classes = 101
            else:
                self.no_classes = 102
        if self.verbose: print 'Test will be CaltechTest'
        
        self.avg_filesize = array([400,400])
        #TODO set alpha value, and implement it for the distance measure
        #TODO performance: mean recognition rate per class
    
    def select_data(self):
        if self.verbose: print 'Selecting Data the Caltech101 way..'
        
        if os.path.exists('/var/scratch/vdvelden/im/caltech101/101_ObjectCategories'):
            # On the nodes, the images should be on the scratch disk
            motherpath = '/var/scratch/vdvelden/im/caltech101/101_ObjectCategories'
        else:
            motherpath = '../im/caltech101/101_ObjectCategories'
        self.classlist = os.listdir(motherpath)
        if self.no_classes < 101:
            random.shuffle(self.classlist)
            self.classlist = self.classlist[:self.no_classes]
        # Select files for each of 3 folders with images, of the sizes indicated
        train_set = []
        test_set = []
        iii=0
        for cls in self.classlist:
            if not (self.difficulty == 'no_background' and cls == 'BACKGROUND_Google'):
                files = self.select_files(motherpath+'/'+cls + '/',  self.trainsize + self.testsize)
                train_set.append(array(files[:self.trainsize]))
                test_set.append(array(files[self.trainsize:]))
        # The train_set is 101D or 102D (101/102 classes), the test_set is 1D (their class doesn't matter) 
        train_set = vstack(train_set)
        test_set  = hstack(test_set)
        # Set the ground truth of the classification of the test images
        self.classification = reshape(tile(range(self.no_classes),[self.testsize,1]), self.no_classes*self.testsize, order='F').astype(uint8)
        print self.classification
        # Return train and test set
        return train_set, test_set
    

class PascalTest(Test):
    """ Class with data_specific functions for a PascalVOC07 test"""
    
    def __init__(self,args):
        super(PascalTest, self).__init__(args,'pascal07')



if __name__ == '__main__':
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--ID', default='1')
    parser.add_argument('TEST')
    parser.add_argument('-c', '--configfile', default="settings.cfg")
    parser.add_argument('-p', '--predict', action='store_true')
    parser.add_argument('-f', '--loadfiles', type=int, default=0 )
    parser.add_argument('-n', '--no_classes', type=int, default=0)
    args = parser.parse_args()
    
    # Depending on the kind of test, initialize a test object accordingly
    if args.TEST == 'graz01_person':
        test = GrazPersonTest(args)
    elif args.TEST == 'graz01_bike':
        test = GrazBikeTest(args)
    elif args.TEST == 'caltech101':
        test = CaltechTest(args)
    elif args.TEST == 'pascal':
        #TODO
        test = PascalTest(args)
    else:
        print 'unknown test: {graz01_person, graz01_bike, caltech101, pascal}'
    
    if args.predict:
        # Only predict the amount of chunks needed to avoid memory problems (size of the training descriptor array),
        # select files accordingly and save the filenames to a file.
        train_files, test_files = test.select_data()
        files_per_chunk = test.predict_descriptor_sizes(test_files.shape[0])
        print 'Files per chunk: ', files_per_chunk
        no_chunks = int(ceil(test_files.shape[0]* 1./files_per_chunk))
        print 'No of chunks:', no_chunks
        
        import cPickle as pkl
        if not os.path.exists(args.ID):
            os.mkdir(args.ID)
        with open(args.ID+'/'+str(no_chunks)+'chunks.pkl','wb') as f:
            for i in range(no_chunks):
                ch_max = min(files_per_chunk*(i+1),test_files.shape[0])
                rng = range(files_per_chunk*i,ch_max)
                print '    Chunk size: ',test_files[rng].shape[0]
                pkl.dump((test_files[rng],test.classification[rng]),f)
        with open(args.ID+'/trainfiles.pkl','wb') as f:
            pkl.dump(train_files,f)
            pkl.dump(no_chunks,f)
        exit(0)
    
    if args.loadfiles > 0:
        # Don't do a full test, only perform a test on the chunk specified by the loadfiles-integer
        # (previously set by a run with the -p flag)
        import cPickle as pkl
        with open(args.ID+'/trainfiles.pkl','rb') as f:
            train_files = pkl.load(f)
            no_chunks = pkl.load(f)
        with open(args.ID+'/'+str(no_chunks)+'chunks.pkl','rb') as f:
            for i in range(args.loadfiles):
                test_files, test.classification = pkl.load(f)
    else:
        # Select names of files used as training and test images
        train_files, test_files = test.select_data()
    print 'No of test files: ', test_files.shape
    print 'No of train files: ', [f.shape for f in train_files]
    # To run ColorDescriptor, first set the variables that need be set for all files
    test.set_descriptor_variables()
    
    # Run the ColorDescriptor for all test files to get their descriptors
    test_descr = test.get_descriptors(test_files, True)
    
    # Initialize the nearest neighbors array and their features array (needed if you want to know to which training feature the test feature was closest within a class)
    nns = zeros([test.no_classes,test_descr.shape[0]], dtype=uint32)
    #features = zeros([test.no_classes, test_descr.shape[0]],dtype=uint32)
    # Iterate over the classes of the training set
    for it, classfiles in enumerate(train_files):
        if test.verbose: print "Class no {0}".format(it)
        # For each class, get the descriptors for the training images
        train_descr = test.get_descriptors(classfiles)
        # Find the nearest neighbors for each test descriptor to the current class descriptors
        _, n = test.find_nn(test_descr, train_descr)
        # Add the results (a feature index and the distance to it) to the final arrays
        nns[it,:] = n
    # Determine a class for each test image using the nearest distances per feature per class
    test.get_class(nns)
    
    # Save the results and settings
    test.save_results(train_files, test_files, nns)
    # Make sure to clean up afterwards
    test.remove_tmpfiles()
    
