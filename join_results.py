import os
from PIL import Image
import numpy as np

# Set the variables
A = 'reconstruction'  # Predefined value for A
B = 'control'
C = 'samples_cfg_scale_9.00'
data_dir = '/proj/control-net-affordance/image_log/train'

# Find the matching images
image_list = os.listdir(data_dir)
matching_images = [image for image in image_list if image.startswith(A) and image.endswith('.png')]

# # Process each matching image
for image in matching_images:
    # Determine the value of X
    X = image.replace(A, '')
    print(X)
    # # Load the images
    image_a = np.array(Image.open(data_dir + '/'+ A + X))
    image_b = np.array(Image.open(data_dir + '/'+ B + X))
    image_c = np.array(Image.open(data_dir + '/'+ C + X))
    
    # Concatenate the images horizontally
    result_array = np.vstack((image_a, image_b, image_c))
    result = Image.fromarray(result_array)
    
    # Save the result as "D_X.png"
    result.save(data_dir + '/joined_'+ X)
