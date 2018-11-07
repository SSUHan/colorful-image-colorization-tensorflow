import os

f = open('data/train.txt', 'w')
imagenet_basepath = './data/train'
for each_image_name in os.listdir(imagenet_basepath):
    image_path = os.path.abspath(os.path.join(imagenet_basepath, each_image_name))
    print(image_path)
    f.write(image_path + '\n')
f.close()
