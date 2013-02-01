from ConfigParser import RawConfigParser
import numpy as np
from nbnn.vocimage import *
from nbnn.voc import *
from nbnn import descriptor, nbnn
from cal import *
import logging

log = logging.getLogger(__name__)



def assert_dir(path):
    if not os.path.exists(path):
        if not '%' in path:
            os.mkdir(path)
        else:
            raise Exception("Trying to init a directory ''%s'', which is not fully formatted"%path)

def get_confidence_values(distances):
    cv = []
    for d in distances:
        no_descriptors = d.shape[0]
        # Sum or Sum of squared?
        cv.append(-d.sum()/no_descriptors)
    return cv

def sort_descriptors(descriptors, images, logger=None):
    if not logger is None:
        log = logger
    else:
        log = logging.getLogger(__name__)
    log.debug('  - Sorting descriptors: %d, imgs:%d',len(descriptors), len(images))
    image_list = [(im_id, p, d) for im_id, (p, d) in descriptors.items()]
    del descriptors
    num_descriptors = [d.shape[0] for i,p,d in image_list]
    descriptors_array = np.vstack([d for i,p,d in image_list])
    points_list = [p for i,p,d in image_list]
    im_ids = [i for i,p,d in image_list]
    imgs = []
    del image_list
    for it1, ref_id in enumerate(im_ids):
        for it2, im in enumerate(images):
            if im.im_id == ref_id:
                # log.debug('  -- descr no %d (id:%s) == list no %d (id:%s)',it1,ref_id,it2,im.im_id)
                imgs.append(im)
                break
    images = imgs
    return descriptors_array, points_list, images, num_descriptors

def getopts(configfile):
    cfg = RawConfigParser()
    cfg.read(configfile)
    
    GLOBopts = dict(cfg.items("GLOBAL"))
    # Set global datatypes:
    if 'nn_threads' in GLOBopts:
        GLOBopts['nn_threads'] = int(GLOBopts['nn_threads'])
    if 'det_threads' in GLOBopts:
        GLOBopts['det_threads'] = int(GLOBopts['det_threads'])
    if 'randbg' in GLOBopts:
        GLOBopts['randbg'] = int(GLOBopts['randbg'])
    
    # Make sure folders exist:
    assert_dir(GLOBopts['tmp_dir'])
    assert_dir(GLOBopts['res_dir'])
    assert_dir(GLOBopts['res_dir']+'/logs')
    assert_dir(GLOBopts['tmp_dir']+'/descriptors')
    assert_dir(GLOBopts['tmp_dir']+'/nbnn')
    
    # Create important paths
    GLOBopts['result_path'] = GLOBopts['res_dir']+'/%s_%s.pkl'
    GLOBopts['log_path'] = GLOBopts['res_dir']+'/logs/%s.log'
    GLOBopts['descriptor_path'] = GLOBopts['tmp_dir']+'/descriptors/%s.dbin'
    GLOBopts['nbnn_path'] = GLOBopts['tmp_dir']+'/nbnn/%s'
    tmpdir = GLOBopts['tmp_dir']
    
    DESCRopts = []
    for i,d in enumerate(["TEST-DESCRIPTOR", "TRAIN-DESCRIPTOR"]):
        # Set datatypes
        ddict = dict(cfg.items(d))
        dtype = ddict['dtype']
        DESCRopts.append((dtype, ddict))
        del DESCRopts[i][1]['dtype']
        if 'outputformat' in DESCRopts[i][1]:
            DESCRopts[i][1]['outputFormat'] = DESCRopts[i][1]['outputformat']
            del DESCRopts[i][1]['outputformat']
        # Set and create paths
        DESCRopts[i][1]['cache_dir'] = '/'.join([GLOBopts['tmp_dir'], DESCRopts[i][1]['cache_dir']])
        assert_dir(DESCRopts[i][1]['cache_dir'])
    
    ndict = dict(cfg.items("NBNN"))
    # Set datatypes
    if 'target_precision' in ndict:
        ndict['target_precision'] = float(ndict['target_precision'])
    if 'checks' in ndict:
        ndict['checks'] = int(ndict['checks'])
    if 'log_level' in ndict:
        ndict['log_level'] = int(ndict['log_level'])
    if ndict['behmo'] == "True":
        mode = 'behmo'
    else:
        mode = 'boiman'
    del ndict['behmo']
    NBNNopts = (mode, ndict)
    
    TESTopts = dict(cfg.items("TEST"))
    # Set datatypes
    if 'batch_size' in TESTopts:
        TESTopts['batch_size'] = int(TESTopts['batch_size'])
    if 'train_size' in TESTopts:
        TESTopts['train_size'] = int(TESTopts['train_size'])
    if 'test_size' in TESTopts:
        TESTopts['test_size'] = int(TESTopts['test_size'])
    if 'k' in TESTopts:
        TESTopts['k'] = int(TESTopts['k'])
    if 'keep_descriptors' in TESTopts:
        TESTopts['keep_descriptors'] = TESTopts['keep_descriptors'] == 'True'
    #  Set Paths
    TESTopts['img_pickle_path'] = '/'.join([tmpdir, TESTopts['img_pickle_path']])
    assert_dir('/'.join(TESTopts['img_pickle_path'].split('/')[:-1]))
    
    # Add the infofile for the test as variable
    TESTopts['infofile'] = '/'.join([tmpdir,'testinfo.txt'])
    
    if cfg.has_section("DETECTION"):
        detdict = dict(cfg.items("DETECTION"))
        method = detdict['method']
        DETopts = (method, detdict)
        # Set datatypes
        if 'theta_m' in DETopts[1]:
            DETopts[1]['theta_m'] = float(DETopts[1]['theta_m'])
        if 'theta_p' in DETopts[1]:
            DETopts[1]['theta_p'] = float(DETopts[1]['theta_p'])
        if 'tau' in DETopts[1]:
            DETopts[1]['tau'] = float(DETopts[1]['tau'])
        else:
            DETopts[1]['tau'] = np.inf
        
        # Set paths
        DETopts[1]['exemplar_path'] = '/'.join([tmpdir, DETopts[1]['exemplar_path']])
        DETopts[1]['distances_path'] = '/'.join([tmpdir, DETopts[1]['distances_path']])
        DETopts[1]['hypotheses_path'] = '/'.join([tmpdir, DETopts[1]['hypotheses_path']])
        exemplar_dir = '/'.join(DETopts[1]['exemplar_path'].split('/')[:-1])
        distances_dir = '/'.join(DETopts[1]['distances_path'].split('/')[:-1])
        hypotheses_dir = '/'.join(DETopts[1]['hypotheses_path'].split('/')[:-1])
        if 'knn_path' in DETopts[1]:
            DETopts[1]['knn_path'] = '/'.join([tmpdir, DETopts[1]['knn_path']])
            distances_dir = '/'.join(DETopts[1]['knn_path'].split('/')[:-1])
            assert_dir(distances_dir)
        assert_dir(exemplar_dir)
        assert_dir(distances_dir)
        assert_dir(hypotheses_dir)
        if 'quickshift_tree_path' in DETopts[1]:
            DETopts[1]['quickshift_tree_path'] = '/'.join([tmpdir, DETopts[1]['quickshift_tree_path']])
            quickshift_tree_dir = '/'.join(DETopts[1]['quickshift_tree_path'].split('/')[:-1])
            assert_dir(quickshift_tree_dir)
        return GLOBopts, DESCRopts, NBNNopts, TESTopts, DETopts
    else:
        return GLOBopts, DESCRopts, NBNNopts, TESTopts, None

def init_descriptor(DESCRopts, logger=None):
    if not logger is None:
        log = logger
    else:
        log = logging.getLogger(__name__)
    log.info(' ++ Initializing descriptor class %s with options: %s', DESCRopts[0], DESCRopts[1].items())
    if DESCRopts[0] == 'DescriptorUint8':
        return descriptor.DescriptorUint8(**DESCRopts[1])
    elif DESCRopts[0] == 'VL_DSift':
        return descriptor.VL_DSift(**DESCRopts[1])
    elif DESCRopts[0] == 'RootSIFT':
        return descriptor.RootSIFT(**DESCRopts[1])
    else:
        raise Exception("Descriptor type '%s' unknown, check your cfg file"%DESCRopts[0])

def init_estimator(path, NBNNopts, logger=None):
    if not logger is None:
        log = logger
    else:
        log = logging.getLogger(__name__)
    log.info(' ++ Initializing estimator class %s (path:%s), with options: %s', NBNNopts[0], path, NBNNopts[1].items())
    if NBNNopts[0] == 'behmo':
        return nbnn.OptNBNNEstimator(path, **NBNNopts[1])
    elif NBNNopts[0] == 'boiman':
        return nbnn.NBNNEstimator(path, **NBNNopts[1])
    else:
        raise Exception("Unknown estimator type '%s', check your cfg file"%NBNNopts[0])


if __name__ == '__main__':
    import doctest
    import numpy as np
    doctest.testmod()