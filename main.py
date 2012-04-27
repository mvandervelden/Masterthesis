from nbnn import *
from caltech import *
from graz import *
from performance import *
import sys, ConfigParser, tarfile, os, shutil

def parse_cfg(cfg_file):
    with open(cfg_file) as cfg:
        test_parameters = dict()
        descriptors = []
        flann_parameters = dict()

        config = ConfigParser.RawConfigParser()
        config.readfp(cfg)
        sections = config.sections()
        for section in sections:
            if section == 'Test':
                for name, value in config.items(section):
                    test_parameters[name] = value
            elif section == 'Flann':
                for name, value in config.items(section):
                    if name=='verbose':
                        flann_parameters[name] = value=='True'
                    else:
                        flann_parameters[name] = value
            else:
                d = [section, dict()]
                for name, value in config.items(section):
                    if name=='alpha':
                        d[1][name] = float(value)
                    elif name=='verbose':
                        d[1][name] = value=='True'
                    else:
                        d[1][name] = value
                descriptors.append(d)
    return test_parameters, descriptors, flann_parameters

if __name__ == "__main__":
    """ Run main using a congif file as argument. The file is parsed into 
        general test parameters, a list of descriptor_args (a dict per 
        descriptor) and flann args. It runs the expected test.
    
    """
    if len(sys.argv) < 2:
        raise Exception("Please give a config file as command line argument")
    
    configfile = sys.argv[1]
    test_parameters, descriptor_args, flann_args = parse_cfg(configfile)

    trainsize = int(test_parameters['trainsize'])
    testsize = int(test_parameters['testsize'])
    testdir = test_parameters['testdir']
    resultsdir = test_parameters['resultdir']
    teststr = test_parameters['test']
    
    if not os.path.exists(resultsdir):
        os.mkdir(resultsdir)
    else:
        raise Exception("Results dir already exists: %s"%resultsdir)
    
    if teststr == 'caltech':
        no_classes = int(test_parameters['no_classes'])
        
        descriptors = [eval(d)(**kwargs) for d,kwargs in descriptor_args]
        test = CaltechTest(testdir, descriptors, trainsize, testsize, 
            no_classes, flann_args)
        result = test.run_test(nbnn_classify)
        print "Caltech Test performed"
    else:
        filetype = test_parameters['filetype']
        difficult = test_parameters['difficult'] == 'True'
        descriptors = [eval(d)(**kwargs) for d,kwargs in descriptor_args]
        test = GrazTest(testdir, descriptors, trainsize, testsize, \
            filetype, teststr, difficult, flann_args)
        result = test.run_test(nbnn_classify)
        print "Graz "+teststr+"Test performed"
    print 'len test:', len(test.test_set)
    print 'len train:', len(test.train_set)
    test_ground_truth = \
        [test.get_ground_truth(image).keys()[0] for image in test.test_set]
    result_classes = [r for (_,r) in result]
    print result_classes
    #print test_ground_truth
    #print result
    with open(resultsdir+'/gt.txt','w') as gt:
        for t in test_ground_truth:
            gt.write(t+'\n')
    with open(resultsdir+'/res.txt','w') as rf:
        for r in result_classes:
            rf.write(r+'\n')
    # Copy the settings file to the resultsfolder for future reference
    shutil.copy(configfile, resultsdir)
    # Copy the result to the resultfolder
    # Using 'with' not possible in python < v2.7
    resulttarfile = resultsdir+'/'+testdir.split('/')[-1]+'.tar.gz'
    
    rtf = tarfile.open(resulttarfile,'w:gz')
    rtf.add(testdir)
    rtf.close()
    shutil.rmtree(testdir)

    cf,class_list = get_confusion_matrix(test_ground_truth, result_classes)
    print cf
    print class_list
    print get_equal_error_rate(cf)
        