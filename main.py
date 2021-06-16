import numpy as np
import tensorflow.keras as keras
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import SGD,Adam,RMSprop
from model import deep_resnet
from data import load_dataset

REQ_W = 224
REQ_H = 224
N_CHANNELS = 3
TRAIN_IMAGES_INPUT_PATH = 'road_segmentation_ideal/training/input/'
TRAIN_IMAGES_OUTPUT_PATH = 'road_segmentation_ideal/training/output/'
BATCH_SIZE = 1
IMG_W = 1500
IMG_H = 1500

def image_reconstructor_out(img_array):
    ref_img = np.zeros((IMG_H,IMG_W))
    indx = 0
    for m in range(6):
        h = [m*250,250+m*250]
        for n in range(6):
            w = [n*250,250+n*250]
            ref_img[h[0]:h[1],w[0]:w[1]] = cv2.resize(img_array[indx,...],(250,250))
            indx+=1
    return ref_img

def image_reconstructor_inp(img_array):
    ref_img = np.zeros((IMG_H,IMG_W,N_CHANNELS))
    indx = 0
    for m in range(6):
        h = [m*250,250+m*250]
        for n in range(6):
            w = [n*250,250+n*250]
            ref_img[h[0]:h[1],w[0]:w[1],:] = cv2.resize(img_array[indx,...],(250,250))
            indx+=1
    return ref_img

class data_generator(keras.utils.Sequence):
    def __init__(self,img_labels,batch_size,shuffle=True,mem_control=True):
        self.img_labels = img_labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mem_control = mem_control

    def __len__(self):
        return int(np.ceil(float(len(self.img_labels))/self.batch_size))
        

    def __getitem__(self,indx):

        # According to the paper since training the image directly on the 
        # entire image itself causes the edges to blur out and hence decrease
        # the accuracy. Thus divide the image into various 224 x 224 images
        # then train the network.
        lbound  = indx*self.batch_size
        upbound = (indx+1)*self.batch_size

        if upbound>len(self.img_labels):
            upbound = len(self.img_labels)
            lbound  = upbound - self.batch_size
        x_train = np.zeros(((upbound-lbound)*36,REQ_H,REQ_W,3))
        y_train = np.zeros(((upbound-lbound)*36,REQ_H,REQ_W))
        indx_1 = 0
        indx_2 = 0
        for i in range(lbound,upbound,1):
            # load the images
            
            im1 = cv2.imread(filename=TRAIN_IMAGES_INPUT_PATH+self.img_labels[i])
            im1 = cv2.cvtColor(im1,cv2.COLOR_BGR2RGB)
            # divide each image into 15 equal images and resize each to
            # required size of the input
            for m in range(6):
                h = [m*250,250+m*250]
                for n in range(6):
                    w = [n*250,250+n*250]
                    im = im1[h[0]:h[1],w[0]:w[1],:]
                    im = cv2.resize(im,(REQ_W,REQ_H))
                    im = im/255.0
                    x_train[indx_1,...] = im
                    indx_1+=1
            im2 = cv2.imread(filename=TRAIN_IMAGES_OUTPUT_PATH+self.img_labels[i])
            im2 = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
            # divide each image into 15 equal images and resize each to
            # required size of the input
            for m in range(6):
                h = [m*250,250+m*250]
                for n in range(6):
                    w = [n*250,250+n*250]
                    im = im2[h[0]:h[1],w[0]:w[1]]
                    im = cv2.resize(im,(REQ_W,REQ_H))
                    im = im/255.0
                    y_train[indx_2,...] = im
                    indx_2+=1
        if self.mem_control:
            return x_train[0:2,...],y_train[0:2,...]
        else:
            return x_train,y_train

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.img_labels)

batch_generator = data_generator(img_labels=load_dataset(),batch_size=BATCH_SIZE,shuffle=True,mem_control=True)
valid_generator = data_generator(img_labels=load_dataset(),batch_size=1,shuffle=True,mem_control=False)
val_generator = data_generator(img_labels=load_dataset(),batch_size=1,shuffle=True,mem_control=False)
x_test,y_test = val_generator.__getitem__(1)
y_pred = np.zeros(shape=y_test.shape)
print(np.max(x_test))
print(np.max(y_test))

DEEP_RESNET = deep_resnet()
DEEP_RESNET.summary()

early_stop = EarlyStopping(monitor='loss',patience=10,mode='min',verbose=1)
checkpoint = ModelCheckpoint('weights_resnet.h5',monitor='loss',verbose=1,
                            save_best_only=True,mode='min',save_freq='epoch')
class PredictionCallback(keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        indx=0
        for layer in range(len(x_test)):
            y_pred[indx,...] = np.reshape(self.model.predict(x_test[layer:layer+1,...]),[224,224])
            indx+=1
                # plot the 3 images to view the result
        print(np.max(y_pred))
        fig = plt.figure(figsize=(8,8))
        plt.subplot(1,3,1)
        plt.title('x_test')
        plt.imshow(image_reconstructor_inp(x_test))

        plt.subplot(1,3,2)
        plt.title('y_test')
        plt.imshow(image_reconstructor_out(y_test),cmap=plt.get_cmap('gray'))

        plt.subplot(1,3,3)
        plt.title('y_pred')
        plt.imshow(image_reconstructor_out(y_pred),cmap=plt.get_cmap('gray'))
        plt.show(block=False)
        plt.pause(3)
        plt.close()

optimizer = SGD(learning_rate=0.001)
DEEP_RESNET.compile(loss='binary_crossentropy',optimizer = optimizer,metrics=[keras.metrics.Precision(),keras.metrics.Recall()])
training_history = DEEP_RESNET.fit(batch_generator,
                                   epochs = 50,
                                   verbose = 1,
                                   callbacks = [early_stop,checkpoint,PredictionCallback()],
                                   )

# load the best model and check prediction
DEEP_RESNET_FINAL = keras.models.load_model('weights_resnet.h5')
val_generator = data_generator(img_labels=load_dataset(),batch_size=1,shuffle=True,mem_control=False)
x_test,y_test = val_generator.__getitem__(1)
y_pred = np.zeros(shape=y_test.shape)
indx=0
for layer in range(len(x_test)):
    print(x_test[layer:layer+1,...].shape)
    y_pred[indx,...] = np.reshape(DEEP_RESNET_FINAL.predict(x_test[layer:layer+1,...]),[224,224])
    indx+=1

print(x_test.shape)
print(y_test.shape)
print(y_pred.shape)
# plot the 3 images to view the result
fig = plt.figure(figsize=(8,8))
plt.subplot(1,3,1)
plt.title('x_test')
plt.imshow(image_reconstructor_inp(x_test))

plt.subplot(1,3,2)
plt.title('y_test')
plt.imshow(image_reconstructor_out(y_test),cmap=plt.get_cmap('gray'))

plt.subplot(1,3,3)
plt.title('y_pred')
plt.imshow(image_reconstructor_out(y_pred),cmap=plt.get_cmap('gray'))
plt.show()