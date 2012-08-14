import numpy as np
import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cPickle, sys

if __name__ == '__main__':
    im_id = sys.argv[1]
    folder = sys.argv[2]
    path = folder+'/distances/%s.pkl'
    exemp_fname = folder+'exemplars/object.npy'
    with open(path%'distances','rb') as f:
        distances = cPickle.load(f)
        allpoints = cPickle.load(f)
        images = cPickle.load(f)
        nearest_exemplar_indexes = cPickle.load(f)
    with open(exemp_fname,'rb') as f:
        exemplars = np.load(f)
    im_it=0
    image = None
    for it,im in enumerate(images):
        if im.im_id == im_id:
            im_it = it
            image = im
            break
    
    fg_distances = distances[im_it][:,0]
    bg_distances = distances[im_it][:,1]
    points = np.asarray(allpoints[im_it])
    
    n = points.shape[0]
    print "showing image %s, with file %s (%d points)"%(im_id, path, n)
    
    imf = Image.open('ex_imgs2/%s.jpg'%im_id)
    imarr = np.asarray(imf)
    xsize = imarr.shape[0]
    plt.imshow(imarr)
    plt.savefig('/'.join(path.split('/')[:-2])+'/'+im_id+'.png')
    plt.clf()
    
    
