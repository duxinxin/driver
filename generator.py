
import keras
import cv2
import numpy as np
import threading
class DXGenerator(keras.utils.Sequence):
	def __init__(self,data,n,des_size=(224,224),means=[103.939, 116.779, 123.68],is_directory=True,batch_size=32,shuffle=True,seed=0):

		super(DXGenerator,self).__init__()
		self.x=data[0]
		self.y=data[1]
		self.n=n
		self.means=means
		self.des_size=des_size
		self.is_directory=is_directory
		self.batch_size=batch_size
		self.shuffle=shuffle
		self.seed=seed
		self.name = None
		self.batch_index=0
		self.total_batches_seen=0
		self.lock=threading.Lock()
		self.index_array=None
		self.index_generator=self._flow_index()

	def reset_index(self):
		self.batch_size=0

	def _set_index_array(self):
		self.index_array=np.arange(self.n)
		if self.shuffle:
			self.index_array=np.random.permutation(self.n)

	def on_epoch_end(self):
		self._set_index_array()

	def __len__(self):
# 		print("len:",(self.n+self.batch_size-1)//self.batch_size)
		return self.n//self.batch_size

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

	def _flow_index(self):
		self.reset_index()
		while True:
# 			print("_flow_index")
			if self.seed is not None:
				np.random.seed(self.seed+self.total_batches_seen)
			if self.batch_size == 0:
				self._set_index_array()

			current_index = (self.batch_index * self.batch_size) % self.n
			if self.n > current_index + self.batch_size:
				self.batch_index += 1 
			else:
				self.batch_index = 0
			self.total_batches_seen += 1

			yield self.index_array[current_index:
									current_index+self.batch_size]
	def __next__(self,*args,**kwargs):
		return self.next()

	def next(self):
# 		print("next")
		with self.lock:
			array = next(self.index_generator)
		if self.seed is not None:
			np.random.seed(self.seed+self.total_batches_seen)
		self.total_batches_seen += 1
# 		print("next1")
		if self.index_array is None:
			self._set_index_array()
		#array = self.index_array[self.batch_size*index:self.batch_size*(index+1)]
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
# 		print("next2")
		return imgs,labels






