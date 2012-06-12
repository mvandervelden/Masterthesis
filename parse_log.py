
import sys,re

if __name__ == '__main__':
    logfile=sys.argv[1]
    total_regex = re.compile('Tot:([0-9]+)')
    swap_regex = re.compile('Swap:([0-9]+)')
    totalmax = [(0,'') for i in range(10)]
    swapmax = [(0,'') for i in range(10)]
    for line in open(logfile):
        totmatch = re.search(total_regex,line)
        swapmatch = re.search(swap_regex,line)
        if totmatch is None:
            continue
        totaluse = int(totmatch.groups()[0])
        swapuse =  int(swapmatch.groups()[0])
        if totaluse > totalmax[0][0]:
            totalmax.pop(0)
            totalmax.append((totaluse,line))
            totalmax.sort()
        if swapuse > swapmax[0][0]:
            swapmax.pop(0)
            swapmax.append((swapuse,line))
            swapmax.sort()
    print [ i for i,j in totalmax]
    for i,j in totalmax:
        print j
    for i,j in swapmax:
        print j