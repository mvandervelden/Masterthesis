from nbnn import *
from caltech import *
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
                d = dict()
                for name, value in config.items(section):
                    if name=='alpha':
                        d[name] = float(value)
                    elif name=='verbose':
                        d[name] = value=='True'
                    else:
                        d[name] = value
                descriptors.append(d)
    return test_parameters, descriptors, flann_parameters

if __name__ == "__main__":
    """ Run main using a congif file as argument. The file is parsed into 
        general test parameters, a list of descriptor_args (a dict per 
        descriptor) and flann args. It runs the expected test.
    
    """
    if len(sys.argv) < 2:
        raise Exception("Please give a config file as command line argument")

    test_parameters, descriptor_args, flann_args = parse_cfg(sys.argv[1])

    trainsize = int(test_parameters['trainsize'])
    testsize = int(test_parameters['testsize'])
    testdir = test_parameters['testdir']
    no_classes = int(test_parameters['no_classes'])
    resultdir = test_parameters['resultdir']

    if test_parameters['test'] == 'caltech':
        descriptors = [XYDescriptor(**kwargs) for kwargs in descriptor_args]
        test = CaltechTest(testdir, descriptors, trainsize, testsize, 
            no_classes, flann_args)
        result = test.run_test(nbnn_classify)
    test_ground_truth = \
        [test.get_ground_truth(image).keys()[0] for image in test.test_set]
    print test_ground_truth
    print result
    with open(resultdir+'/gt.txt','w') as gt:
        for t in test_ground_truth:
            gt.write(t+'\n')
    with open(resultdir+'/res.txt','w') as rf:
        for r in result:
            rf.write(r+'\n')
    # Copy the result to the resultfolder
        resulttarfile = resultdir+'/'+testdir.split('/')[-1]+'.tar.gz'
    with tarfile.open(resulttarfile,'w:gz') as rtf:
        rtf.add(testdir)
    shutil.rmtree(testdir)
        