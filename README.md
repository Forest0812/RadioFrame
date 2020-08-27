

## 项目框架

```
.
├── checkpoint
│   └── Mnist_LeNet_V1				#dir，存储Mnist数据集下LeNet的模型和日志文件
├── dataLoader						#dir，存储数据集处理函数
│   ├── __init__.py
│   ├── MNIST.py					#Mnist数据及文件
├── metric							#dir，存储各类的metric
│   ├── classifyMetric.py			#分类有关的metric
│   ├── __init__.py
├── model							#dir，存储各类model
│   ├── __init__.py
│   ├── lenet.py					#LeNet模型，用于Mnist训练
├── trainer							#dir，模型训练器
│   ├── BaseTrainer.py				#所有训练器的父类，存储一些共同操作
│   ├── __init__.py
│   ├── LeNetTrainer.py				#LeNet该网络对应的训练器
├── train.py						#用于模型的训练
├── test.py							#用于模型的测试
├── config.py						#配置文件
└── utils.py						#功能函数

```



## Train

train之前，要依次对GPU、数据集、训练模型、损失函数、优化方法、metrics、tricks进行设置

### 1) GPU

在`config.GPU`中进行设置`

- `use_gpu`： True表示采用GPU
- `device_id`：所采用的GPU的设备号



### 2) 数据集

以Mnist为例

**选择框架存在的数据集**

1. 在`config.CONFIG['dataset_name']`中设置数据集名称

**添加框架之外的数据集**

1. 在`dataLoader`文件夹中写好数据加载函数，`MnistTrainSet、MnistTestSet`
2. 在`config`中添加数据集参数（数据加载函数所需要的参数）

```
self.Mnist = dict(dirname='',)  ##注意config中的数据集名称要与dataLoader的数据类名称相同
```

3. 在`config.CONFIG['dataset_name']`中设置所选的数据集名称



### 3) 训练模型

以LeNet为例

1. 在`model`文件夹中添加模型类LeNet
2. 在`config`中添加模型类参数（loadLeNet所需要的参数）

```
self.LeNet = dict()  ##注意config中的模型类名称要与model的模型类名称相同
```

3. 在`config.CONFIG['model_name']`中设置所选的模型名称



### 4) 损失函数

**选择`torch.nn`中的损失函数**

1. 在`config.CONFIG['criterion_name']`中设置所选的损失函数名称CrossEntropyLoss



### 5) 优化方法

**选择`torch.nn`中的优化方法**

1. 在`config.CONFIG['optimizer_name']`中设置所选的损失函数名称Adam
2. 在`config`中添加优化方法参数（优化方法所需要的参数）

```
self.Adam = dict(
            lr = 0.01,                  #学习率
            weight_decay = 5e-4,        #权重衰减
        )
```



### 6) Metrics

1. 在`metric`文件夹中写好相应的metric
2. 在`self.config.CONFIG['metrics']`中添加所需要的`metric`，注意：这里`metircs`为列表，可以选择多个`metirc`，而且`metric`函数的输入目前只支持两个参数`logits`和`target`



### 7) 自适应修改学习率

1. 在`self.config.CONFIG['adjust_lr']`中设置，`True` 代表采用自适应修改学习率
2. 目前支持的修改学习率方法只有随`epoch`变更学习率的方法
3. 在`self.LRAdjust`中设置相应的参数



### 8) 加载预训练模型

注意：该选项也应用在`test.py`的应用中

1. 在`self.config.CONFIG['load_model']`中设置，`True` 代表加载预训练模型
2. 在`self.config.LoadModel`中设置预训练模型的位置（注意：预训练模型所采用的模型要与`self.config.CONFIG['model_name']`是一样的）



### 9) 开始训练

```
python train.py
```





## Test

test之前，要依次对GPU、数据集、训练模型、metrics进行设置

### 1) GPU

在`config.GPU`中进行设置`

- `use_gpu`： True表示采用GPU
- `device_id`：所采用的GPU的设备号



### 2) 数据集

以Mnist为例

**选择框架存在的数据集**

1. 在`config.CONFIG['dataset_name']`中设置数据集名称

**添加框架之外的数据集**

1. 在`dataLoader`文件夹中写好数据加载函数，`MnistTrainSet、MnistTestSet`
2. 在`config`中添加数据集参数（数据加载函数所需要的参数）

```
self.Mnist = dict(dirname='',)  ##注意config中的数据集名称要与dataLoader的数据类名称相同
```

3. 在`config.CONFIG['dataset_name']`中设置所选的数据集名称



### 3) 训练模型

以LeNet为例

1. 在`model`文件夹中添加模型类LeNet
2. 在`config`中添加模型类参数（loadLeNet所需要的参数）

```
self.LeNet = dict()  ##注意config中的模型类名称要与model的模型类名称相同
```

3. 在`config.CONFIG['model_name']`中设置所选的模型名称



### 4) Metrics

1. 在`metric`文件夹中写好相应的metric
2. 在`self.config.CONFIG['metrics']`中添加所需要的`metric`，注意：这里`metircs`为列表，可以选择多个`metirc`，而且`metric`函数的输入目前只支持两个参数`logits`和`target`



### 5) 加载预训练模型

注意：该选项也应用在`test.py`的应用中

1. 在`self.config.CONFIG['load_model']`中设置，`True` 代表加载预训练模型
2. 在`self.config.LoadModel`中设置预训练模型的位置（注意：预训练模型所采用的模型要与`self.config.CONFIG['model_name']`是一样的）



### 9) 开始测试

```
python test.py
```





