## 数据标注极限的实验

### 比例实验

#### 比例 

0，1-pixel，10-pixel， 1‰， 5‰， 1%  	5%  	10%  	50%  100%

#### model-（pretrained or not）

- vgg16-FCN
- mobilenet-fcn
- resnet18-FCN
- dilation-resnet18-FCN
- JPU（resnet18）
- PSP-net（resnet101）
- non-local（resnet18）
- mobilenet
- densenet


判断稀疏性标注的来源可能是哪儿
- 标签本身的标记是冗余的（基于感受野的分析）
- pretrain model使得模型具有了一定的特征提取能力
    - 3部分，没有pretrain，imagenet，具体数据集， 逐层
- 模型特征表达的的冗余性

然后的实验就是，
- 不同的结构设计是否会降低对于标注数据的要求，卷积神经网络的局部相关与nonlocal的全局相关性
- 如果数据降低的情况下，哪些标注点更重要（TODO）

### 关键点实验

#### PV-Net 实验

#### key-annotation 实验





