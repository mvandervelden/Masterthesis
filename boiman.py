# encoding: utf-8
"""
boiman.py

Created by Maarten van der Velden on 2012-02-21.
Copyright (c) 2012. All rights reserved.
"""

import os
import re
import numpy as np
import DescriptorIO, pyflann

class Parameters():
    pass

def select_data(test, params):
    trainsz = params.trainsize
    testsz  = params.testsize
    diff = params.diff
    if test == 'graz01_person':
        pathp = "../im/graz01/persons/"
        pathb = "../im/graz01/bikes/"
        pathn = "../im/graz01/no_bike_no_person/"
        
        pos_files = np.array(select_files(pathp,  trainsz + testsz, diff))
        bike_files= np.array(select_files(pathb, (trainsz + testsz)/2, diff))
        no_files  = np.array(select_files(pathn, (trainsz + testsz)/2, diff))
        train_set = np.vstack([pos_files[:trainsz], np.hstack([bike_files[:trainsz/2],no_files[:trainsz/2]])])
        #[pos_files[:trainsz], bike_files[:trainsz/2] + no_files[:trainsz/2]]
        test_set  = np.hstack([pos_files[trainsz:],bike_files[trainsz/2:], no_files[trainsz/2:]])
        #pos_files[trainsz:] + bike_files[trainsz/2:] + no_files[trainsz/2:]
        classification = np.array([0]*testsz + [1]*testsz, int)
        return train_set, test_set, classification
    elif test == 'graz01_bike':
        pass
    elif test == 'caltech101':
        pass
    elif test == 'graz01_descriptors':
        pathp = "../im/graz01/persons/"
        pathb = "../im/graz01/bikes/"
        pathn = "../im/graz01/no_bike_no_person/"
        pers_files = np.array(select_files(pathp,  0, diff))
        bike_files= np.array(select_files(pathb, 0, diff))
        no_files  = np.array(select_files(pathn, 0, diff))

        fileset  = np.hstack([pers_files,bike_files, no_files])
        print fileset.shape
        return fileset

def select_files(path, sz, diff):
    files = os.listdir(path)
    def filt(x): return re.search('.jpg',x)
    files = filter(filt,files)

    if diff == 'no_hard':
        if 'graz01/persons' in path:
            limit = 348
        elif 'graz01/bikes' in path:
            limit = 306
        elif 'graz01/no_bike_no_person' in path:
            limit = 273
        def rm_hard(x, limit): return int(re.search('[0-9]+',x).group(0)) <= limit
        files = [f for f in files if rm_hard(f, limit)]

    if sz == 0:
        files = [path+f for f in files]
        return files
    np.random.shuffle(files)
    files = files[:sz]
    files = [path+f for f in files]
    return files

def get_descriptors(files, params, tmp_dir):
    from subprocess import Popen, PIPE
    no_files = len(files)
    if not tmp_dir[:-1] in os.listdir('.'):
        os.mkdir(tmp_dir)
    outputbase = params.detector + "_" + params.descriptor + "_"
    if params.binary:
        dopts = ["--detector", params.detector, "--descriptor", params.descriptor, "--outputFormat", 'binary',"--output"]
        outputext = '.dbin'
    else:
        dopts = ["--detector", params.detector, "--descriptor", params.descriptor,"--output"]
        outputext = '.dtxt'
    
    descriptors = [np.array(0) for i in range(no_files*params.scale_levels)]
    
    if params.scale_levels > 1:
        import cv2.cv as cv
    
    for i, f in enumerate(files):
        print "generating descriptors for {0}".format(f)
        filename = f.split('/')[-1]
        if not params.remove_tmp:
            tmp_dir = f[:-len(filename)] + 'descriptors/'
            if not 'descriptors' in os.listdir(f[:-len(filename)]):
                os.mkdir(tmp_dir)
                print 'tmp_dir: ',tmp_dir
        o = tmp_dir+outputbase+filename[:-4]+outputext
        run_args= ['colorDescriptor', f] + dopts + [o]
        print o
        #print run_args
        #print type(run_args)
        p = Popen(run_args,stdout=PIPE, stderr=PIPE)
        _,err = p.communicate()
        if not err == '':
            raise Exception("ColDescriptor run failed.\n Message: {0}\n Did not make output for {1}".format(err, f))
        else:
            if params.remove_tmp:
                # read descriptors:
                _, descriptors[i*params.scale_levels] = DescriptorIO.readDescriptors(o)
                #print [d.shape for d in descriptors]
                os.remove(o)
        
        if params.scale_levels > 1:
            print "  scaling:",
            filebase = filename[:-4]
            filext = filename[-4:]
            fnew = f
            im1 = cv.LoadImageM(fnew)
            for scale in range(1,params.scale_levels):
                print scale,
                im2 = cv.CreateMat(im1.rows/2,im1.cols/2,im1.type)
                cv.PyrDown(im1, im2)
                filename = filebase+'_'+str(scale)+filext
                fnew = tmp_dir+filename
                cv.SaveImage(fnew,im2)
                o = tmp_dir+outputbase+filebase+'_scale'+str(scale)+outputext
                run_args= ['colorDescriptor', fnew] + dopts + [o]
                #print run_args
                #print type(run_args)
                p = Popen(run_args,stdout=PIPE, stderr=PIPE)
                _,err = p.communicate()
                if not err == '':
                    raise Exception("ColDescriptor run failed.\n Message: {0}\n Did not make output for {1}".format(err, fnew))
                else:
                    if params.remove_tmp:
                        # read descriptors:
                        _, descriptors[i*params.scale_levels+scale] = DescriptorIO.readDescriptors(o)
                        #print [d.shape for d in descriptors]
                        os.remove(o)
                os.remove(fnew)
                im1 = im2
            print '' 
    dssize = np.array([d.shape[0] for d in descriptors]).reshape(params.scale_levels,no_files).sum(0)
    descriptors = np.vstack(descriptors)
    return descriptors, dssize

def find_nn(train_descr, test_descr, params):
    flann = pyflann.FLANN()
    _, dists = flann.nn(train_descr, test_descr, num_neighbors=params.k, \
            algorithm='kdtree')
    return np.array(dists)
   
def get_class(nns, dssize, no_files):
    # nns is a n x m array, where n=number of classes, m=numbr of test descriptors
    # dssize = no of features per image
    # testsz = no of test images
    starts = np.hstack([0, np.cumsum(dssize)])
    #print dssize, starts,nns.shape
    c_hat = np.zeros(no_files,int)
    for i in xrange(no_files):
        nfile = nns[starts[i]:starts[i+1],:]
        nsum = np.sum(nfile**2,0)
        c_hat[i] = np.argmin(nsum)
    return c_hat

def get_performance(c, ch, params):
    if params.verbose:
        print 'Classes       : ', c
        print 'Classification: ', ch
    
    sumdirection = 1
    if len(c.shape) == 1:
        sumdirection = 0
    
    tp = ((c==ch) & (c==0)).sum(sumdirection).mean()
    tn = ((c==ch) & (c==1)).sum(sumdirection).mean()
    fp = ((c!=ch) & (c==1)).sum(sumdirection).mean()
    fn = ((c!=ch) & (c==0)).sum(sumdirection).mean()
    
    if params.verbose:
    
        print ''
        print 'Confusion matrix: predicted:'
        print ' actual :       : {0}   {1}'.format(tp, fn)
        print '                  {0}   {1}'.format(fp, tn)
    
    f = open(params.resultsfile,'w')
    
    f.write('[Parameters]\n')
    for k,v in vars(params).items():
        f.write('{0}: {1}\n'.format(k,v))
        if isinstance(v, Parameters):
            for kk,vv in vars(v).items():
                f.write('  {0}: {1}\n'.format(kk,vv))
    f.write('\n[Results]\n')
    f.write('Truth:          {0}\n'.format(c))
    f.write('Classification: {0}\n\n'.format(ch))
    f.write('conf: | cht | chf \n')
    f.write('  ----+-----|-----\n')
    f.write('   ct |{0:4.1f} |{1:4.1f} \n'.format(tp,fn))
    f.write('   cf |{0:4.1f} |{1:4.1f} \n'.format(fp,tn))
    f.close()
    
if __name__ == '__main__':
    import ConfigParser, argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--ID', default='1')
    parser.add_argument('-c', '--configfile', default="settings.cfg")
    parser.add_argument('-s', '--descriptoronly', action='store_true')
    args = parser.parse_args()
   
    config = ConfigParser.RawConfigParser()
    config.read(args.configfile)
    
    params = Parameters()
    params.ID = args.ID    
    params.verbose = config.getboolean('General', 'verbose')
    r = config.get('General', 'resultsfile').split('.')
    params.resultsfile = r[0]+'_'+params.ID+'.'+r[1]
    params.tmpdir = 'tmp_'+args.ID+'/'
    params.iterations = config.getint('General', 'iterations')
    params.test = config.get('Data', 'test')
    
    params.data = Parameters()
    params.data.trainsize = config.getint('Data','trainsize')
    params.data.testsize = config.getint('Data', 'testsize')
    params.data.diff = config.get('Data', 'difficulty')
    
    params.descr = Parameters()
    params.descr.descriptor = config.get('Descriptors', 'descriptor')
    params.descr.detector = config.get('Descriptors', 'detector')
    params.descr.binary = config.getboolean('Descriptors', 'save_binary')
    params.descr.scale_levels = config.getint('Descriptors', 'scale_levels')
    params.descr.remove_tmp = True
    
    params.flann = Parameters()
    params.flann.k = config.getint('Flann', 'k')
    
    if args.descriptoronly:
        params.test = 'graz01_descriptors'
        params.ID = ''
        params.descr.remove_tmp = False
        if params.verbose:
            print params.test
        
        filenames = select_data(params.test, params.data)
        get_descriptors(filenames, params.descr, params.tmpdir)
    else:
        if params.verbose:
            print params.test
    
        c_hats = np.array(0)
        classes = np.array(0)
    
        for it in range(params.iterations):
            trainfiles, testfiles, classification = select_data(params.test, params.data)
            no_classes = trainfiles.shape[0]
            if classes.shape == ():
                c_hats = np.zeros([params.iterations, no_classes*params.data.testsize])
                classes = np.zeros([params.iterations, no_classes*params.data.testsize])
            print 'Getting test descriptors'
            test_descr, dssize = get_descriptors(testfiles, params.descr, params.tmpdir)
            print '\n'
            nns = np.zeros([test_descr.shape[0],no_classes])
            for i,clf in enumerate(trainfiles):
                print 'Get training descriptors for class {0}'.format(i)
                train_descr, _ = get_descriptors(clf, params.descr, params.tempfile)
                print '  Find NN'
                #idxes = make_indexes(train_descr, params.flann)
                nns[:,i] = find_nn(train_descr, test_descr, params.flann)
            print 'Calculate c_hat'
            c_hat = get_class(nns, dssize, params.data.testsize*no_classes)
            classes[it,:] = classification
            c_hats[it,:] = c_hat
        get_performance(classes, c_hats, params)
    
    