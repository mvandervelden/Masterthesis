import cPickle, sys
import numpy as np
from logutils import *
from utils import *
from file_io import *
from detection_utils import *
from metric_functions import *
from nbnn.voc import *

if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise Exception("arguments expected: cfgfile class im_id")
    configfile = sys.argv[1]
    cls = sys.argv[2]
    # Get options into dicts
    VOCopts = VOC.fromConfig(configfile)
    GLOBopts, DESCRopts, NBNNopts, TESTopts, DETopts = getopts(configfile)
    DETopts = DETopts[1]
    # Setup logger
    log = init_log(GLOBopts['log_path'], 'ranking_%s'%cls, 'w')
    
    # logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    # log = logging.getLogger(__name__)
    
    log.info("Making VOC results files cfg:%s, cls:%s",configfile, cls)
    
    outputf = GLOBopts['res_dir']+'/comp3_det_%s_%s.txt'%(GLOBopts['test_set'], cls)
    
    vimages = read_image_set(VOCopts, GLOBopts['test_set'])
    log.info('Ranking images: %s',' '.join([im.im_id for im in vimages]))
    all_detections = []
    all_det_vals = []
    all_det_imids = []
    for vimage in vimages:
        im_id = vimage.im_id
        log.info('Parsing image %s detections...', im_id)
        detfile = GLOBopts['result_path']%(im_id, cls)
    
        detections, reflist, distances, points = load_detections(detfile,im_id)
        if detections.shape[0] == 0:
            log.info("No detections for image %s, skip this image",im_id)
            continue
        if not isinstance(reflist[0], np.ndarray):
            # If reflist is a lst of lists instead of a list of ndarrays, convert
            reflist = [np.array(l) for l in reflist]
        log.info(" Detections: %s, Reflist: %s (max: %d), distances: %s, points: %s", detections.shape, len(reflist), max([l.max() for l in reflist]), distances.shape, points.shape)
        detection_vals = get_detection_values(detections, reflist, distances, points, eval(DETopts['detection_metric']))
        log.info("im %s: det shape=%s, det_vals shape=%s"%(im_id, detections.shape, detection_vals.shape))
        all_detections.append(detections)
        all_det_vals.append(detection_vals)
        imids = np.array([im_id for i in range(detections.shape[0])])
        all_det_imids.append(imids)
        log.info("stored imids shape:%s", imids.shape)
    all_detections = np.vstack(all_detections)
    all_det_vals = np.vstack(all_det_vals)
    all_det_imids = np.hstack(all_det_imids)
    log.info("Found %s detections, %s vals, %s imids", all_detections.shape, all_det_vals.shape, all_det_imids.shape)
    ranking = sort_values(all_det_vals)
    log.info("ranking shape: %s", ranking.shape)
    
    save_voc_results(outputf, all_detections[ranking], all_det_vals[ranking], all_det_imids[ranking])
    
    log.info('FINISHED')
