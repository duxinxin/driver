import keras 
import threading
import cv2
import numpy as np
class TestGenerator(keras.utils.Sequence):
    def __init__(self,data,n,des_size=(224,224),means=[103.939, 116.779, 123.68],is_directory=True,batch_size=32,seed=0):
        '''
        data: tuple of (x,y)
        n: data size
        des_size: standard size
        means: the dataset mean of RGB,default is imagenet means [103.939, 116.779, 123.68]
        batch_size: default is 32
        shuffle: random the data,default is True
        seed: the random seed,default is 0
        '''
        super(TestGenerator,self).__init__()
        self.data = data
        self.n = n
        self.means = means
        self.des_size = des_size
        self.is_directory = is_directory
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_array = None
        self.index_generator = self._flow_index()
        
    def reset_index():
        self.batch_index = 0
    
    def _set_index_array(self):
        self.index_array = np.arange(self.n)
        if self.shuffle:
            self.index_array = np.random.permutation(self.n)
            
    def on_epoch_end():
        self._set_index_array()
    
    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size  # round up
    # keras will call this function for data if the class is subclass of Sequence,otherwise will call the next
    def __getitem__(self, index):
        # random choose the index
        if self.seed is not None:
            # affect the np.random.permutation,random generate the index for array
            np.random.seed(self.seed + self.total_batches_seen)
        self.total_batches_seen += 1
        # 
        if self.index_array is None:
            self._set_index_array()
        # request the memory
        index_array = self.index_array[self.batch_size*index:self.batch_size(index+1)]
        imgs = np.zeros((len(index_array),self.des_size[0],self.des_size[1],3))
        # read the data
        if self.is_directory:
            #print(self.x)
            # read from path
            img_names=self.data[index_array]
            for name_index in range(len(img_names)):
                #print(index,self.batch_size)
                img = cv2.imread(img_names[name_index])
                if img is not None:
                #print(img)
                    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                #print(self.des_size)
                    img = cv2.resize(img,self.des_size)
                    imgs[name_index] = img 
               
        else:
            for i in range(len(index)):
                img = self.data[index[i]]
                img = cv2.resize(img,self.des_size)
                imgs[i] = img 

        # standardlize
        # transform to -1~1
#         x = imgs - self.means
        return imgs
            
    def _flow_index(self):
        # data generate
        # ensure the batch_index is 0
        self.reset_index()
        while True:
            if self.seed is not None:
                np.random.seed(self.seed + self.total_batches_seen)
            if self.batch_index == 0:
                self._set_index_array()

            current_index = (self.batch_index * self.batch_size) % self.n
            # batch_index will be set 0 when the value * batch_size large to the n
            if self.n > current_index + self.batch_size:
                self.batch_index += 1
            else:
                self.batch_index = 0
            self.total_batches_seen += 1
            
            yield self.index_array[current_index:
                                   current_index + self.batch_size]
    # in python3, __next,2 is next
    def __next__(self, *args, **kwargs):
        return self.next()
    
    def next(self):
        with self.lock:
            #index_array = next(self.index_generator)
            self.batch_index += 1
        return self.__getitem__(self.batch_index)
    