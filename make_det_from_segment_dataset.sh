S_TRAINSET=$(cat VOCdevkit/VOC2007/ImageSets/Segmentation/train.txt)
S_VALSET=$(cat VOCdevkit/VOC2007/ImageSets/Segmentation/val.txt)
CLASSES="aeroplane bicycle bird boat bottle bus car cat chair cow diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor"
echo $S_TRAINSET
for cls in $CLASSES; do
    filename_t="VOCdevkit/VOC2007/ImageSets/Main/${cls}_segtrain.txt"
    rm $filename_t
    echo $filename_t
    for a in $S_TRAINSET; do
        cat "VOCdevkit/VOC2007/ImageSets/Main/${cls}_trainval.txt" | awk -v nn=${a} '$1 ~ nn {print $0}' >> ${filename_t}
    done
    
    filename_v="VOCdevkit/VOC2007/ImageSets/Main/${cls}_segval.txt"
    rm $filename_v
    for a in $S_VALSET; do
        cat "VOCdevkit/VOC2007/ImageSets/Main/${cls}_trainval.txt" | awk -v nn=${a} '$1 ~ nn {print $0}' >> ${filename_v}
    done
    
    filename_tv="VOCdevkit/VOC2007/ImageSets/Main/${cls}_segtrainval.txt"
    echo filename_tv
    rm $filename_tv
    cat ${filename_t} ${filename_v} > $filename_tv
done