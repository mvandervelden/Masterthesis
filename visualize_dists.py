import numpy as np
import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cPickle, sys

if __name__ == '__main__':
    path = sys.argv[1]
    with open(path,'rb') as f:
        distances = cPickle.load(f)
        allpoints = cPickle.load(f)
        im_ids = cPickle.load(f)
        im_objs = cPickle.load(f)
    
    fg_distances = distances[0][:,0]
    bg_distances = distances[0][:,1]
    points = np.asarray(allpoints[0])
    im_id = im_ids[0]
    for img in im_objs:
        if img.im_id == im_id:
            image = img
            break
    
    xsize = image.width
    
    n = points.shape[0]
    print "showing image %s, with file %s (%d points)"%(im_id, path, n)
    print "image size: (%d,%d), from points(%d,%d)"%(image.width,image.height,points[:,0].max(), points[:,1].max() )
    
    imf = Image.open(image.path)
    imarr = np.asarray(imf)
    if False:
    
        for i,scale in enumerate([1.33, 2., 3., 4.5]):
            print "plotting descriptor scale %f"%scale
            pts = points[i:n:4,:]
            fg = fg_distances[i:n:4]
            bg = bg_distances[i:n:4]
            print "mindist, maxdist to fg: ",fg.min(),',',fg.max()
            print "mindist, maxdist to bg: ",bg.min(),',',bg.max()

            plt.subplot(121)
            plt.imshow(imarr)
            plt.hexbin(pts[:,0], pts[:,1],C=fg,alpha=.2,gridsize=xsize/8 )
            plt.subplot(122)
            plt.imshow(imarr)
            plt.hexbin(pts[:,0], pts[:,1],C=bg,alpha=.2,gridsize=xsize/8 )
        
            plt.savefig(path[:-4]+'_%.2f.png'%scale)
            plt.clf()
        # Plot averaged distances
        plt.subplot(121)
        plt.imshow(imarr)
        plt.hexbin(points[:,0], points[:,1],C=fg_distances,alpha=.2,gridsize=xsize/8 )
        plt.subplot(122)
        plt.imshow(imarr)
        plt.hexbin(points[:,0], points[:,1],C=bg_distances,alpha=.2,gridsize=xsize/8 )
        plt.savefig(path[:-4]+'_summedscales.png')
        plt.clf()
    
        # Plot normalized distances, to compare
        maxd = max([fg_distances.max(), bg_distances.max()])
        norm_fgd = fg_distances/maxd
        norm_bgd = bg_distances/maxd
    
        plt.subplot(121)
        plt.imshow(imarr)
        plt.hexbin(points[:,0], points[:,1],C=norm_fgd,vmin=0,vmax=1,alpha=.2,gridsize=xsize/8 )
        plt.subplot(122)
        plt.imshow(imarr)
        plt.hexbin(points[:,0], points[:,1],C=norm_bgd,vmin=0,vmax=1,alpha=.2,gridsize=xsize/8 )
        plt.savefig(path[:-4]+'_norm_scales.png')
        plt.clf()
    
        # plot dist difference, to compare: redder = closer to bg, bluer = closer to fg
        diff_d = fg_distances-bg_distances
        plt.imshow(imarr)
        plt.hexbin(points[:,0], points[:,1],C=diff_d,alpha=.2,gridsize=xsize/8 )
        plt.savefig(path[:-4]+'_diff.png')
        plt.clf()
    
    for i,im_id in enumerate(im_ids):
        print im_id
        points = np.asarray(allpoints[i])
        fg_distances = distances[i][:,0]
        bg_distances = distances[i][:,1]
        for img in im_objs:
            if img.im_id == im_id:
                imarr = np.asarray(Image.open(img.path))
                break
        
        # plot dist ratio, to compare: redder = closer to bg, bluer = closer to fg
        diff_d = (bg_distances-fg_distances)/fg_distances
        plt.imshow(imarr)
        plt.hexbin(points[:,0], points[:,1],C=diff_d,alpha=.2,gridsize=xsize/8 ,bins='log')
        cb = plt.colorbar()
        cb.set_label('log10((bgd-fgd)/fgd)')
        plt.savefig(path[:-4]+'_%s_quality.png'%im_id)
        plt.clf()
    
    