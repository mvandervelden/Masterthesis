import numpy as np
import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cPickle, sys

if __name__ == '__main__':
    
    d_path = sys.argv[1]
    im_path = sys.argv[2]
    im_id = sys.argv[3]
    with open(d_path%'distances','rb') as f:
        distances = cPickle.load(f)
        allpoints = cPickle.load(f)
        images = cPickle.load(f)
        nearest_exemplar_indexes = cPickle.load(f)
        
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
    print "showing image %s (%d points)"%(im_id, n)
    
    imf = Image.open(im_path%im_id)
    imarr = np.asarray(imf)
    xsize = imarr.shape[0]
    plt.imshow(imarr)
    plt.savefig('/'.join(d_path.split('/')[:-2])+'/'+im_id+'.png')
    plt.clf()
    print "image size: %s, from points(%d,%d)"%(imarr.shape,points[:,0].max(), points[:,1].max() )
    if False:
    
        for i,scale in enumerate([1.33, 2., 3., 4.5]):
            print "plotting descriptor scale %f"%scale
            pts = points[i:n:4,:]
            fg = fg_distances[i:n:4]
            bg = bg_distances[i:n:4]
            print "mindist, maxdist to fg: ",fg.min(),',',fg.max()
            print "mindist, maxdist to bg: ",bg.min(),',',bg.max()

            plt.subplot(121)
            # plt.imshow(imarr)
            plt.hexbin(pts[:,0], pts[:,1],C=fg,alpha=.2,gridsize=xsize/5 )
            cb = plt.colorbar()
            plt.subplot(122)
            # plt.imshow(imarr)
            plt.hexbin(pts[:,0], pts[:,1],C=bg,alpha=.2,gridsize=xsize/5 )
            cb = plt.colorbar()
        
            plt.savefig('/'.join(d_path.split('/')[:-2])+'/'+im_id+'_%.2f.png'%scale)
            plt.clf()
        # Plot averaged distances
        plt.subplot(121)
        # plt.imshow(imarr)
        plt.hexbin(points[:,0], points[:,1],C=fg_distances,alpha=.2,gridsize=xsize/10 )
        plt.subplot(122)
        # plt.imshow(imarr)
        plt.hexbin(points[:,0], points[:,1],C=bg_distances,alpha=.2,gridsize=xsize/10 )
        plt.savefig('/'.join(d_path.split('/')[:-2])+'/'+im_id+'_summedscales.png')
        plt.clf()
    
        # Plot normalized distances, to compare
        maxd = max([fg_distances.max(), bg_distances.max()])
        norm_fgd = fg_distances/maxd
        norm_bgd = bg_distances/maxd
    
        plt.subplot(121)
        # plt.imshow(imarr)
        plt.hexbin(points[:,0], points[:,1],C=norm_fgd,vmin=0,vmax=1,alpha=.2,gridsize=xsize/10 )
        plt.subplot(122)
        # plt.imshow(imarr)
        plt.hexbin(points[:,0], points[:,1],C=norm_bgd,vmin=0,vmax=1,alpha=.2,gridsize=xsize/10 )
        plt.savefig('/'.join(d_path.split('/')[:-2])+'/'+im_id+'_norm_scales.png')
        plt.clf()
    
        # plot dist difference, to compare: redder = closer to bg, bluer = closer to fg
        diff_d = fg_distances-bg_distances
        # plt.imshow(imarr)
        plt.hexbin(points[:,0], points[:,1],C=diff_d,alpha=.2,gridsize=xsize/10 )
        plt.savefig('/'.join(d_path.split('/')[:-2])+'/'+im_id+'_diff.png')
        plt.clf()
    

    # plot dist ratio, to compare: redder = closer to bg, bluer = closer to fg
    # diff_d = (bg_distances-fg_distances)/fg_distances
    diff_d = bg_distances-fg_distances
    plt.imshow(imarr)
    plt.hexbin(points[:,0], points[:,1],C=diff_d,alpha=.2,gridsize=xsize/10)
    # ,bins='log')
    cb = plt.colorbar()
    cb.set_label('bg_dist - fg_dist')
    plt.savefig('/'.join(d_path.split('/')[:-2])+'/'+im_id+'_quality_im.png')
    plt.clf()
    
    # plot dist ratio, to compare: redder = closer to bg, bluer = closer to fg
    plt.hexbin(points[:,0], points[:,1],C=diff_d,alpha=.2,gridsize=xsize/10)
    # ,bins='log')
    cb = plt.colorbar()
    cb.set_label('bg_dist - fg_dist')
    plt.savefig('/'.join(d_path.split('/')[:-2])+'/'+im_id+'_quality.png')
    plt.clf()
    
    
    