from visualize import *


if __name__ == "__main__":
    try:
        method = sys.argv[1]
        cfgfile = sys.argv[2]
    except IndexError:
        print "Not enough command line arguments:"
        print usage
        exit(1)
    
    if len(sys.argv) > 3:
        options = sys.argv[3:]
    else:
        options = None
    
    GLOBopts, DESCRopts, NBNNopts, TESTopts, DETopts = getopts(cfgfile)
    