import sys
import numpy as np
import matplotlib.pyplot as plt

def get_equal_error_rate(confmat):
    """ Calculate the n-D ROC Equal Error Rate of a confusion matrix"""
    
    # Get the prior probability of each class by dividing the amount of images
    # in the class by the total
    ssums = confmat.sum(1)
    priors = ssums/ssums.sum()
    # Get the amount of "true positives" for each class (the diagonal of the CF)
    corrects = np.diag(confmat)
    # Get the true positive rate for each class by dividing the amounts of true
    # positives by their amounts
    truth_rates = corrects/ssums
    # The sum of true positive rates times their prior over the classes defines
    # the equal error rate (with 2 classes, this is p1*tpr + p2*(1-fpr)[=tnr])
    return (truth_rates*priors).sum()

if __name__ == '__main__':
    resultsfile = sys.argv[1]
    filename = '.'.join(resultsfile.split('.')[:-1])
    with open(resultsfile, 'r') as f:
        content = f.read()
    rows = content.split('\n')
    confmat = np.zeros([len(rows)-1,len(rows)-1])
    for i,row in enumerate(rows):
        strvals = row.split(' ')
        if not len(strvals) == len(rows):
            continue
        rvals = np.zeros(len(rows)-1)
        for j,elem in enumerate(strvals):
            if not elem == '':
                rvals[j] = int(elem)
        confmat[i]= rvals

    eer = get_equal_error_rate(confmat)
    print "Mean Recognition Rate: ", eer
    trues = confmat.sum(0)
    norm_cf = confmat/trues
    
    
    classes = np.asarray(["Faces","Faces_easy","Leopards",\
        "Motorbikes","accordion","airplanes","anchor","ant","barrel",\
        "bass","beaver","binocular","bonsai","brain","brontosaurus",\
        "buddha","butterfly","camera","cannon","car_side","ceiling_fan",\
        "cellphone","chair","chandelier","cougar_body","cougar_face","crab",\
        "crayfish","crocodile","crocodile_head","cup","dalmatian",\
        "dollar_bill","dolphin","dragonfly","electric_guitar","elephant",\
        "emu","euphonium","ewer","ferry","flamingo","flamingo_head",\
        "garfield","gerenuk","gramophone","grand_piano","hawksbill",\
        "headphone","hedgehog","helicopter","ibis","inline_skate",\
        "joshua_tree","kangaroo","ketch","lamp","laptop","llama","lobster",\
        "lotus","mandolin","mayfly","menorah","metronome","minaret",\
        "nautilus","octopus","okapi","pagoda","panda","pigeon","pizza",\
        "platypus","pyramid","revolver","rhino","rooster","saxophone",\
        "schooner","scissors","scorpion","sea_horse","snoopy","soccer_ball",\
        "stapler","starfish","stegosaurus","stop_sign","strawberry",\
        "sunflower","tick","trilobite","umbrella","watch","water_lilly",\
        "wheelchair","wild_cat","windsor_chair","wrench","yin_yang"])

    norm_tp = np.diag(norm_cf)
    tp_idxs = np.argsort(norm_tp)
    fiveworst = classes[tp_idxs[:10]]
    worstvals = norm_tp[tp_idxs[:10]]
    print "The 5 worst classes: ", zip(fiveworst, worstvals)
    fivebest = classes[tp_idxs[-10:]]
    bestvals = norm_tp[tp_idxs[-10:]]
    print "The 5 best classes: ", zip(fivebest, bestvals)
    
    
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    # ax.set_aspect(1)
    res = ax.imshow(norm_cf, cmap=plt.cm.jet, 
                    interpolation='nearest')
    
    width = confmat.shape[0]
    height = confmat.shape[1]

    # for x in xrange(width):
    #      for y in xrange(height):
    #          ax.annotate(str(confmat[x, y]), xy=(y, x), 
    #                      horizontalalignment='center',
    #                      verticalalignment='center')
    # 
    cb = fig.colorbar(res)
    # alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    # plt.xticks(range(width), alphabet[:width])
    # plt.yticks(range(height), alphabet[:height])
    plt.savefig(filename+'.png', format='png')
    print "Confusion matrix saved to "+filename+".png"
