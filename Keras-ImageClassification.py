
# coding: utf-8

# In[1]:


from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:





# In[3]:


train_img = sorted(list(paths.list_images('dataset/train')))
valid_img = sorted(list(paths.list_images('dataset/test')))


# In[4]:


train_data = []
valid_data = []
train_labels = []
valid_labels = []


# In[5]:


for imagePath in train_img:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (28, 28))
    image = img_to_array(image)
    train_data.append(image)
    label = imagePath.split(os.path.sep)[-2]
    if label=='cats':
        label=0
        train_labels.append(label)
    elif label=='dogs':
        label=1
        train_labels.append(label)
    else:
        label=2
        train_labels.append(label)


# In[6]:


for imagePath in valid_img:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (28, 28))
    image = img_to_array(image)
    valid_data.append(image)
    label = imagePath.split(os.path.sep)[-2]
    if label=='cats':
        label=0
        valid_labels.append(label)
    elif label=='dogs':
        label=1
        valid_labels.append(label)
    else:
        label=2
        valid_labels.append(label)


# In[7]:


# print(len(train_data),len(valid_data))


# In[8]:


# print(len(train_labels),len(valid_labels))


# In[9]:


train_data = np.array(train_data,dtype='float32') / 255.0
valid_data = np.array(valid_data,dtype='float32') / 255.0


# In[10]:


train_labels = np.array(train_labels)
valid_labels = np.array(valid_labels)


# In[11]:


from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers.core import Activation

clf= Sequential()

clf.add(Conv2D(32, (5, 5), input_shape = (28, 28, 3), activation = 'relu'))

clf.add(MaxPooling2D(pool_size = (2, 2)))

clf.add(Conv2D(32, (3, 3), activation = 'relu'))
clf.add(MaxPooling2D(pool_size = (2, 2)))

clf.add(Conv2D(64, (3, 3), activation = 'relu'))
clf.add(MaxPooling2D(pool_size = (2, 2)))

clf.add(Flatten())
clf.add(Dense(500))
clf.add(Activation("relu"))

clf.add(Dense(3))
clf.add(Activation("sigmoid"))
clf.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[12]:


# clf.summary()


# In[13]:


train_labels=to_categorical(train_labels, num_classes=3)
valid_labels=to_categorical(valid_labels, num_classes=3)


# In[106]:


aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest")

clf.compile(loss="binary_crossentropy", optimizer='adam',
    metrics=["accuracy"])

clf.fit_generator(aug.flow(train_data,train_labels, batch_size=20),
validation_data=(valid_data,valid_labels), steps_per_epoch=(train_data.shape[0])//20,
epochs=30,verbose=1)


# In[ ]:


cd test


# In[108]:


ls


# In[109]:


import imutils


image = cv2.imread(test+'/sel.jpg')

orig = image.copy()
image = cv2.resize(image, (28, 28))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)


# In[110]:


(cat, dog,selena)=clf.predict(image)[0]


# In[111]:


if ((cat>dog) and (cat>selena)):
    label='cat'
    proba=cat*100[:3]
elif ((dog>cat) and (dog>selena)):
    label='dog'
    proba=dog*100
else:
    label='selena'
    proba=selena*100    


# In[112]:


text=label+ ' ' +str(proba)[:5]


# In[113]:


output = imutils.resize(orig, width=400)
cv2.putText(output, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)


# In[114]:


plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))


# In[115]:


image = cv2.imread(test+'/ca.jpg')

orig = image.copy()
image = cv2.resize(image, (28, 28))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)


# In[116]:


(cat, dog,selena)=clf.predict(image)[0]


# In[118]:


if ((cat>dog) and (cat>selena)):
    label='cat'
    proba=cat*100
elif ((dog>cat) and (dog>selena)):
    label='dog'
    proba=dog*100
else:
    label='selena'
    proba=selena*100    


# In[119]:


text=label+ ' ' +str(proba)[:5]


# In[120]:


output = imutils.resize(orig, width=400)
cv2.putText(output, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)


# In[121]:


plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))


# In[123]:


image = cv2.imread(test+'/dogg.jpg')

orig = image.copy()
image = cv2.resize(image, (28, 28))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)


# In[124]:


(cat, dog,selena)=clf.predict(image)[0]


# In[125]:


if ((cat>dog) and (cat>selena)):
    label='cat'
    proba=cat*100[:3]
elif ((dog>cat) and (dog>selena)):
    label='dog'
    proba=dog*100
else:
    label='selena'
    proba=selena*100    


# In[126]:


text=label+ ' ' +str(proba)[:5]


# In[127]:


output = imutils.resize(orig, width=400)
cv2.putText(output, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0,0,255), 2)


# In[128]:


plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))

