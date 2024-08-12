import numpy as np 
import cv2
from cv2 import dnn
import matplotlib.pyplot as plt
 
#--------Model file paths--------#
proto_file = 'colorization_deploy_v2.prototxt'
model_file = 'colorization_release_v2.caffemodel'
hull_pts = 'pts_in_hull.npy'
img_path = 'img-1.jpg'
#--------------#--------------#
 
#--------Reading the model params--------#
net = dnn.readNetFromCaffe(proto_file, model_file)
kernel = np.load(hull_pts)
#-----------------------------------#---------------------#
 
#-----Reading and preprocessing image--------#
img = cv2.imread(img_path)
scaled = img.astype("float32") / 255.0
lab_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
#-----------------------------------#---------------------#
 
# Add the cluster centers as 1x1 convolutions to the model
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = kernel.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
#-----------------------------------#---------------------#
 
# We'll resize the image for the network
resized = cv2.resize(lab_img, (224, 224))
# Split the L channel
L = cv2.split(resized)[0]
# Mean subtraction
L -= 50
#-----------------------------------#---------------------#
 
# Predicting the ab channels from the input L channel
net.setInput(cv2.dnn.blobFromImage(L))
ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))
# Resize the predicted 'ab' volume to the same dimensions as our input image
ab_channel = cv2.resize(ab_channel, (img.shape[1], img.shape[0]))
 
# Take the L channel from the image
L = cv2.split(lab_img)[0]
# Join the L channel with predicted ab channel
colorized = np.concatenate((L[:, :, np.newaxis], ab_channel), axis=2)
 
# Then convert the image from Lab to BGR 
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized = np.clip(colorized, 0, 1)
 
# Change the image to 0-255 range and convert it from float32 to int
colorized = (255 * colorized).astype("uint8")
 
# Resize the images and prepare them for display
img_resized = cv2.resize(img, (640, 640))
colorized_resized = cv2.resize(colorized, (640, 640))
 
# Concatenate the original grayscale and colorized images
result = cv2.hconcat([img_resized, colorized_resized])
 
# Convert the result from BGR to RGB for displaying with Matplotlib
result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

# Display the result using Matplotlib
plt.imshow(result_rgb)
plt.title("Grayscale -> Colour")
plt.axis('off')  # Hide axes for better visualization
plt.show()