import tensorflow as tf
from utils import *
from net import Net
from skimage.io import imsave
from skimage.transform import resize
import cv2

IMAGE_HW = 256

data_ph = tf.placeholder(tf.float32, shape=(1, IMAGE_HW, IMAGE_HW, 1))
autocolor = Net(train=False)

conv8_313 = autocolor.inference(data_ph)

saver = tf.train.Saver()
with tf.Session() as sess:
  saver.restore(sess, 'models/model.ckpt')
  
  for i in range(1, 4):
  	print("i : {}".format(i))
  	img = cv2.imread('gray_{}.jpg'.format(i))
	
	if len(img.shape) == 3:
	  img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	# print("img shape : {}".format(img.shape))
	img = cv2.resize(img, (IMAGE_HW, IMAGE_HW))
	# print("img shape 2: {}".format(img.shape))
	img = img[None, :, :, None]
	data_l = (img.astype(dtype=np.float32)) / 255.0 * 100 - 50

  	conv8_313_out = sess.run(conv8_313, feed_dict={data_ph: data_l})

	img_rgb = decode(data_l, conv8_313_out, 2.63)
	imsave('color_{}.jpg'.format(i), img_rgb)
