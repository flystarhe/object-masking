# git
```
git config --global user.name "flystarhe"
git config --global user.email "flystarhe@qq.com"

git init
```

### add submodule
```
git submodule add https://github.com/flystarhe/pyhej.git modules/pyhej

git submodule add https://github.com/mrgloom/awesome-semantic-segmentation.git modules/awesome_semantic_segmentation
git submodule add https://github.com/matterport/Mask_RCNN.git modules/mask_rcnn

git submodule add https://github.com/caocuong0306/awesome-object-proposals.git modules/awesome_object_proposals
git submodule add https://github.com/fizyr/keras-retinanet.git modules/keras_retinanet
git submodule add https://github.com/c0nn3r/RetinaNet.git modules/pytorch_retinanet
git submodule add https://github.com/longcw/yolo2-pytorch.git modules/pytorch_yolo2

https://github.com/amdegroot/ssd.pytorch.git
https://github.com/orobix/retina-unet.git
https://github.com/ycszen/pytorch-seg.git
https://github.com/zhixuhao/unet.git
https://github.com/ZijunDeng/pytorch-semantic-segmentation.git
https://github.com/meetshah1995/pytorch-semseg.git
```

### push
```
git remote add origin git@github.com:flystarhe/object-masking.git
git submodule update --remote
git add .
git commit -m "init"
git push origin master
```

### pull
```
git clone https://***.git
git submodule init
git submodule update

git pull origin master
git submodule update --remote
```
