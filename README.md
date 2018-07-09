<<<<<<< HEAD
# driver_state_check
司机驾驶状态检测
=======
### 1、获取数据集
这个项目为kaggle上的一个竞赛项目，可以从kaggle官网下载[数据集](https://www.kaggle.com/c/state-farm-distracted-driver-detection/data)

### 2、项目描述
这个项目是为了检测司机是否存在不安全的驾驶行为，数据集中的图片是安装在仪表盘上的摄像头拍摄所取。

这个项目的本质是一个监督分类学习的问题，通过输入已经标好标签的训练集进行学习，使得训练好的模型在输入一张新的图片时能准确地判断出图中的司机是否是安全驾驶状态。

### 3、数据集
此项目的类别分为十类(c0-c9)：

c0:安全驾驶

c1:右手打字

c2:右手打电话

c3:左手打字

c4:左手打电话

c5:调收音机

c6:喝饮料

c7:取后座上的物品

c8:整理头发和化妆

c9:与其他人交流

数据集中的每张图片都是640*480的宽高，颜色通道为RGB

### 4、解决方法
使用openCV读取图片，使用numpy进行数据的提取和转换，使用tensorflow进行网络的搭建和训练

### 5、评估
项目是一个多分类的问题，主要有精确度和时间的标准

通过损失函数（多分类交叉熵）来描述模型的性能，损失函数的值越小，模型的性能越好
```math
logloss = -\frac{1}{N}\sum_{i=1}^N\sum_{j=1}^My_{ij}log(p_{ij})
```
p:predict,y:test,i:(1,2,3,4,5,6,...,N(测试集的大小),j:1,2,3,4,5,6,..,M(分类数量))

训练时间长一点，但是预测时长一定要短，因为如果项目投入实际使用，预测时间过长则无意义。

预测时间在0.5秒之内的话可以追求提高准确率，若预测时间大于0.5秒的话则优先考虑时长。因为人也是需要一定的反应时间的，所以如果预测时间较长而给予的反应时间太短的话无法做出有效的行为动作。

# 设计
### 提取数据
由于数据是从视频一帧一帧取得的图片，所以除了在不同动作切换时的前后两帧图片不同时，其他时刻的前后两帧图片是相同的，所以训练集和验证集需要用司机的ID来划分，否则造成的后果是验证集数据将会是模型之前见过的数据，这样的模型将无意义，不便于调参。

```
import numpy as np
import os
import csv
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import keras
from sklearn.cross_validation import train_test_split
from generator import DXGenerator
from testGenerator import TestGenerator

#获取数据
def get_data():
    reader = pd.read_csv('driver_imgs_list.csv')

    driver_ids =np.array(reader['subject'])
    driver_classes = reader['classname']
    driver_imgs = reader['img']

    driver_id_dict={}
    
    for index in range(len(driver_ids)):
        if driver_ids[index] in driver_id_dict.keys():
            driver_id_dict[driver_ids[index]].append((driver_classes[index], driver_imgs[index]))
        else:
            driver_id_dict[driver_ids[index]] = [(driver_classes[index], driver_imgs[index])]

    
    #train
    train_img_name=[]
    train_labels_count=0
    id_key_list = list(driver_id_dict.keys())
    np.random.shuffle(id_key_list)
    for index in id_key_list[:-2]:
        train_labels_count += len(driver_id_dict[index])
    
    train_dict = {}
    train_image_labels = np.zeros((train_labels_count,),dtype=np.int)
#    print(len(train_image_labels))
    for index in id_key_list[:-2]:
        if index in driver_id_dict.keys():
            for i in range(len(driver_id_dict[index])):
                img_name = 'train' + '//' + str((driver_id_dict[index][i])[0]) + '//' + str((driver_id_dict[index][i])[1])
                train_dict[img_name] = int((driver_id_dict[index][i])[0][1:])
    
    for i in range(len(list(train_dict.items()))):
        train_img_name.append(list(train_dict.items())[i][0])
        train_image_labels[i] = list(train_dict.items())[i][1]
#    print(len(train_image_labels))
    
    train = list(zip(train_img_name,train_image_labels))
    train_img_name=[]
    np.random.shuffle(train)
    for i in range(len(train)):
        train_img_name.append(train[i][0])
        train_image_labels[i] = train[i][1]
#    print(len(train_image_labels))
        
    
    #test
    valid_img_name=[]
    valid_labels_count=0
    for index in id_key_list[-2:]:
        valid_labels_count += len(driver_id_dict[index])
    
    valid_image_labels = np.zeros((valid_labels_count,),dtype=np.int)
    valid_dict={}
    for index in id_key_list[-2:]:

        if index in driver_id_dict.keys():
            for i in range(len(driver_id_dict[index])):
                img_name = 'train' + '//' + str((driver_id_dict[index][i])[0]) + '//' + str((driver_id_dict[index][i])[1])
                valid_dict[img_name]=int((driver_id_dict[index][i])[0][1:])

    for i in  range(len(list(valid_dict.items()))):
        valid_img_name.append(list(valid_dict.items())[i][0])
        valid_image_labels[i]=list(valid_dict.items())[i][1]
```
### Generator
由于数据集比较大，所以如果一次性的读取很多图片的话就会导致出现OOM问题，所以要定义一个generator。我是仿照keras的generator进行了自定义，主要继承了keras.utils.Sequence，重写了__getitem__方法，这是数据的产生。

```
def __getitem__(self,index):
# 		print("index==",index)       
		if self.seed is not None:
			np.random.seed(self.seed+self.total_batches_seen)
		self.total_batches_seen += 1

		if self.index_array is None:
			self._set_index_array()
		array = self.index_array[self.batch_size*index:self.batch_size*(index+1)]
		imgs = np.zeros((len(array),self.des_size[0],self.des_size[1],3))
		img_names=[]
		img_labels=[]
		if self.is_directory:

			for i in array:
# 				print(len(self.x))
				img_names.append(self.x[i])
# # 				if self.y is not None:
				img_labels.append(self.y[i])
# 			print(img_names)
			for name_index in range(len(img_names)):
				img = cv2.imread(img_names[name_index])
# 				print(img)
				if img is not None: 
					img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
					img = cv2.resize(img,self.des_size)
					imgs[name_index]=img
# 					img_labels[name_index]=self.y[name_index]
# 			print(len(labels))
		else:
			for i in range(len(index)):
				img = self.x[index[i]]
				img = cv2.resize(img,self.des_size)
				imgs[i] = img
				labels[i] = self.y[index[i]]

# 		print(img_names,img_labels)
		del img_names
		labels = keras.utils.to_categorical(img_labels,10)
		
		return imgs,labels

```
### 迁移学习
在这个项目中，我使用了迁移学习，主要是使用了resnet50、inceptionV3、xception三个模型。将模型在imagenet上训练后，再用这个模型在自己的数据集上进行训练，进行微调，使模型更加拟合我的数据，在数据上表现的更好。这里我的得分最高是90%，这里以resnet50为例

```
#Resnet50
train_generator = DXGenerator((train_imgs,train_labels),len(train_imgs),
                               is_directory=True,batch_size=batch_size,
                               shuffle=True,seed=0)
train_generator.name="train"

valid_generator = DXGenerator((valid_imgs,valid_labels),len(valid_imgs),
                               is_directory=True,batch_size=batch_size,
                               shuffle=True,seed=0)
valid_generator.name="valid"


Input = keras.layers.Input(shape=(224,224,3))
process_input = keras.layers.Lambda(res_preprocess_input)(Input)
resnet_50 = resnet50.ResNet50(include_top=True, weights='imagenet',input_tensor=process_input,pooling='avg')
# resnet_50.summary()
resModel = Model(inputs=resnet_50.input,outputs=resnet_50.layers[-2].output)
#函数式模型
output = resModel.output

#为了防止过拟合，加入dropout层和正则化
output = Dropout(0.6)(output)
#,kernel_regularizer=regularizers.l2(0.01)
predictions = Dense(10,activation='softmax',name='driver_classfier')(output)
# print(predictions)
model = Model(inputs=resModel.input,outputs=predictions)

#先使用resnet最后一层进行训练，防止权重波动太大
for layers in resnet_50.layers:
    layers.trainable=False

sgd = keras.optimizers.SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False)
adam = keras.optimizers.Adam(lr=0.001)

model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit_generator(train_generator,len(train_imgs) // batch_size,
                    epochs=1,verbose=1,
                    validation_data=valid_generator,
                    validation_steps=len(valid_imgs)// batch_size)

#然后放开resnet所有层进行训练
for layers in resModel.layers:
    layers.trainable=True
    
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model_check=ModelCheckpoint("resnet50.h5", monitor='val_loss', 
                            save_best_only=True, save_weights_only=False, mode='auto', period=1)

model.fit_generator(train_generator,len(train_imgs) // batch_size,
                    epochs=10,verbose=1,
                    validation_data=valid_generator,
                    validation_steps=len(valid_imgs) // batch_size,
                   callbacks=[model_check])
```
### 预测
用训练好的模型在测试集上进行预测。
```
test_imgs = load_test()
Epoch 
test_generator=TestGenerator(test_imgs,len(test_imgs) // batch_size,
                             is_directory=True,batch_size=batch_size,
                             seed=0)
result = model.predict_generator(test_generator,len(test_imgs)//batch_size,verbose=1)
print(result)
```

### 总结
该项目计算消耗资源较多,采用了迁移学习的方式，极大的减少了训练时长，在深度网络中需要的数据集是海量的，迁移学习也能有效的避免数据集不足的问题。
>>>>>>> driver_check
