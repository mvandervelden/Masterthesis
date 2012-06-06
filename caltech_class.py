from nbnn import *
from pascal import *
import logging,logging.config
from memuse import *
import os,sys, ConfigParser, shutil

def parse_cfg(cfg_file):
    with open(cfg_file) as cfg:
        test_parameters = dict()
        data_args = dict()
        descriptors = []
        flann_args = dict()
        

        config = ConfigParser.RawConfigParser()
        config.readfp(cfg)
        sections = config.sections()
        for section in sections:
            if section == 'Test':
                for name, value in config.items(section):
                    if name == 'batch_size':
                        if value == 'None':
                            test_parameters[name] = None
                        else:
                            test_parameters[name] = int(value)
                    else:
                        test_parameters[name] = value
            elif section == 'Dataset':
                for name, value in config.items(section):
                    if name == 'classes':
                        data_args[name] = value.split(',')
                    elif name == 'trainsize' or name == 'testsize' or \
                            name == 'no_classes':
                        data_args[name] = int(value)
                    else:
                        data_args[name] = value
            elif section == 'Flann':
                for name, value in config.items(section):
                    if name in ['k','checks','trees']:
                        flann_args[name] = int(value)
                    else:
                        flann_args[name] = value
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
    return test_parameters, data_args, descriptors, flann_args

if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        raise Exception("Please give a config file as command line argument")    
    configfile = sys.argv[1]
    test_params, data_args, descriptor_args, flann_args = parse_cfg(configfile)

    logging.config.fileConfig(test_params['log_config'],disable_existing_loggers=False)
    log = logging.getLogger('')
    f = MemuseFilter()
    log.handlers[0].addFilter(f)

    log.info('================CALTECH101 CLASSIFICATION================')
    log.info('=========================================================')
    log.info('===================INIT CAL101 DATASET===================')
    dataset = CaltechClassification(**data_args)
    log.info("===================INIT RESULTSHANDLER===================")
    vrh = CaltechResultsHandler(dataset,test_params['result_path'])
    log.info('=====================INIT DESCRIPTOR=====================')
    descriptors = [eval(d)(**kwargs) for d,kwargs in descriptor_args]
    log.info('=====================INIT ESTIMATOR =====================')
    estimators = [NBNNEstimator.from_dataset(test_params['temp_path'], dataset, \
        descriptor, **flann_args) for descriptor in descriptors]
    log.info("======================STARTING TEST======================")
    run_test(dataset, descriptors, estimators, vrh.set_results, \
        batch_size=test_params['batch_size'], output_function=nbnn_classify)
    log.info(vrh)
    log.info("=====================SAVING RESULTS =====================")
    vrh.save_to_files()
    log.info("=======================CLEANING UP=======================")
    shutil.rmtree(test_params['temp_path'])
    
