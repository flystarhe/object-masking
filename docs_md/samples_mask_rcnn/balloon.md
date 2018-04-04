# 建立一个颜色填充过滤器
和大多数图像编辑app中包含的过滤器不同,我们的过滤器更加智能一些:它能自动找到目标.当你希望把它应用到视频上而不是图像上时,这种技术更加有用.

https://github.com/flystarhe/object-masking/tree/master/docs/samples_mask_rcnn

## 训练数据集
通常我会从寻找包含所需目标的公开数据集开始.但在这个案例中,我想向你展示这个项目的构建过程,因此我将介绍如何从零开始构建一个数据集.

我在flickr上搜索气球图片,并选取了75张图片,将它们分成了训练集和验证集.找到图片很容易,但标注阶段才是困难的部分.

等等,我们不是需要数百万张图片来训练深度学习模型吗?实际上,有时候需要,有时候则不需要.我是考虑到以下两点而显著地减小了训练集的规模:

- 首先,迁移学习.简单来说,与其从零开始训练一个新模型,我从已在COCO数据集(在repo中已提供下载)上训练好的权重文件开始.虽然COCO数剧集不包含气球类别,但它包含了大量其它图像(约12万张),因此训练好的图像已经包含了自然图像中的大量常见特征,这些特征很有用
- 其次,由于这里展示的应用案例很简单,我并不需要令这个模型达到很高的准确率,小的数据集已足够

有很多工具可以用来标注图像.由于其简单性我最终使用了VIA(VGG图像标注器).这是一个HTML文件,你可以下载并在浏览器中打开.标注最初几张图像时比较慢,不过一旦熟悉了用户界面,就能达到一分钟一个目标的速度.

## 加载数据集
分割掩码的保存格式并没有统一的标准.有些数据集中以PNG图像保存,其它以多边形点保存等.为了处理这些案例,在我们的实现中提供了一个Dataset类,你可以通过重写几个函数来读取任意格式的图像.

VIA工具将标注保存为JSON文件,每个掩码都是一系列多边形点.

BalloonDataset类:
```python
class BalloonDataset(utils.Dataset):

    def load_balloon(self, dataset_dir, subset):
        ...

    def load_mask(self, image_id):
        ...

    def image_reference(self, image_id):
        ...
```

- `load_balloon`读取JSON文件,提取标注,然后迭代地调用内部的`add_class`和`add_image`函数来构建数据集
- `load_mask`通过画出多边形为图像中的每个目标生成位图掩码
- `image_reference`返回鉴别图像的字符串结果,以进行调试,这里返回的是图像文件的路径

你可能已经注意到我的类不包含加载图像或返回边框的函数.基础的Dataset类中默认的`load_image`函数可以用于加载图像,边框是通过掩码动态地生成的.

## 验证该数据集
上文提到的数据可以从这里下载:

- [balloon_dataset.zip](https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip)
- [mask_rcnn_balloon.h5](https://github.com/matterport/Mask_RCNN/releases/download/v2.1/mask_rcnn_balloon.h5)

为了验证我的新代码可以正确地实现,我添加了这个Jupyter notebook:[balloon_inspect_data.ipynb](#),它加载了数据集,并可视化了掩码,边框,还可视化了anchor来验证anchor的大小是否拟合了目标大小.

## 配置
这个项目的配置和训练COCO数据集的基础配置很相似,因此我只需要修改3个值.正如我对Dataset类所设置的,我复制了基础的Config类,然后添加了我的覆写:
```python
class BalloonConfig(Config):
    # Give the configuration a recognizable name
    NAME = "balloon"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + baloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
```

基础的配置使用的是`1024x1024`的输入图像尺寸以获得最高的准确率.我保持了相同的配置,虽然图像相对较小,但模型可以自动地将它们重新缩放.

## 训练
Mask R-CNN是一个规模很大的模型.尤其是在我们的实现中使用了`ResNet101`和`FPN`,因此你需要一个12GB显存的GPU才能训练这个模型.在小规模的数据集上,训练时间不到1个小时.

用以下命令开始训练,从`balloon`的目录开始运行.这里,我们需要指出训练过程应该从预训练的COCO权重开始.代码将从我们的repo中自动下载权重.
```bash
python balloon.py train --dataset=/path/to/dataset --model=coco
```

如果训练停止了,用以下命令让训练继续:
```bash
python balloon.py train --dataset=/path/to/dataset --model=last
```

也可以从预训练的ImageNet权重开始:
```bash
python balloon.py train --dataset=/path/to/dataset --weights=imagenet
```

除了`balloon.py`以外,该repo还有两个例子:`train_shapes.ipynb`,它训练了一个小规模模型来检测几何形状;`coco.py`,它是在COCO数据集上训练的.

## 检查结果
[balloon_inspect_model.ipynb](#)展示了由训练好的模型生成的结果.查看该notebook可以获得更多的可视化选项,并一步一步检查检测流程.(是`inspect_model.ipynb`的简化版本,包含可视化选项和对COCO数据集代码的调试)

## 颜色填充
现在我们已经得到了目标掩码,让我们将它们应用于颜色填充效果.方法很简单:创建一个图像的灰度版本,然后在目标掩码区域,将原始图像的颜色像素复制上去.应用填充效果的代码在`color_splash()`函数中.`detect_and_color_splash()`可以实现加载图像,运行实例分割和应用颜色填充过滤器的完整流程.

Apply splash effect on an image:
```
python balloon.py splash --weights=/path/to/mask_rcnn_balloon.h5 --image=<file name or URL>
```

Apply splash effect on a video. Requires OpenCV 3.2+:
```
python balloon.py splash --weights=/path/to/mask_rcnn_balloon.h5 --video=<file name or URL>
```
