import numpy as np
import os
import Image
import matplotlib.pyplot as plt
import cPickle
import copy

if __name__ == '__main__':
    files = os.listdir('quickshift')
    parentses = []
    distses = []
    sortdistses = []
    idxes = []
    for file in files:
        with open('quickshift/'+file, 'rb') as f:
            p = cPickle.load(f)
            d = cPickle.load(f)
        parentses.append(p)
        distses.append(d)
        idxes.append(range(p.shape[0]))
        sortdistses.append(np.sort(d))
    # for dists in sortdistses:
        # plt.plot(dists)
    # plt.show()
    
    infs = []
    for d in distses:
        infs.append(np.isinf(d).sum())
        print np.sort(d)[-10:]
    selfparents = []
    for p, i in zip(parentses, idxes):
        selfparents.append(np.sum(p==i))
    # print infs
    # print p
    
    taus = [0,1,1.01,1.02,1.03,1.04,1.05,1.06,1.07,1.08,1.09,1.1,1.2,1.25,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,2.5,3,3.5,4,4.5,5]
    
    ax = plt.subplot(1,1,1)
    lalala = []
    for n, dists in enumerate(distses):
        no_clusters = []
        for tau in taus:
            clusters = copy.copy(parentses[n])
            for i, d in enumerate(dists):
                if d <= tau:
                    dd = d
                    p = clusters[i]
                    while dd <= tau:
                        p = clusters[p]
                        dd = dists[p]
                    clusters[i] = p
            n_c = np.unique(clusters).shape[0]
            no_clusters.append(n_c)
        print n
        no_clusters.append(selfparents[n])
        lalala.append(no_clusters)
        # ax.plot(taus+[6], no_clusters)
    # ax.set_yscale('log')
    # plt.show()
    
    print taus[13]
    t125 = [n[13] for n in lalala]
    print t125
    print 'mean', np.mean(t125)
    print 'median', np.median(t125)
    print 'min', np.min(t125)
    print 'max', np.max(t125)