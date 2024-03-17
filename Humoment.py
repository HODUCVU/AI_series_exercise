import numpy as np
import cv2
from PIL import Image
def Image_to_grid(image):
    # Open image, convert to black and white mode
    image = Image.fromarray(image).convert('1')
    w, h = image.size
    # Temporary NumPy array of type bool to work on
    temp = np.array(image, dtype=bool)
    # Detect changes between neighboring pixels
    diff_y = np.abs(np.diff(temp, axis=0).astype(bool))  # Convert diff_y to bool
    diff_x = np.abs(np.diff(temp, axis=1).astype(bool))  # Convert diff_x to bool
    # Create a union image of detected changes
    temp = np.zeros_like(temp, dtype=bool)
    temp[:h-1, :] |= diff_y
    temp[:, :w-1] |= diff_x
    # Calculate distances between detected changes
    diff_y = np.diff(np.nonzero(np.diff(np.sum(temp, axis=0))))
    diff_x = np.diff(np.nonzero(np.diff(np.sum(temp, axis=1))))
    # Calculate tile height and width
    ht = np.median(diff_y[diff_y > 1]) + 2
    wt = np.median(diff_x[diff_x > 1]) + 2
    # Resize image w.r.t. tile height and width
    if (ht > 0 and wt > 0):  # Ensure that ht and wt are greater than 0
        array = np.array(image.resize((int(w/wt), int(h/ht)))).astype(int)
    return array, int(w/wt), int(h/ht)
    
def convert_to_binary(img_grayscale, thresh=80):
    thresh, img_binary = cv2.threshold(img_grayscale, thresh, maxval=255, type=cv2.THRESH_BINARY)
    return img_binary

def Humoment(grid, h, w):    
    # m00
    m00 = np.sum(grid)
    xs =0
    ys = 0
    centroid = np.array([0.0, 0.0])
    for y in range(0, int(h)):
        for x in range(0, int(w)):     
            if grid[y][x] == 1:
                ys += y + 1
                xs += x + 1 
                centroid += np.array([y + 1, x + 1])
    print('ys = ', ys, ' xs = ', xs)
    centroid /= float(m00)
    print("m00 =", m00)
    print("centroidx =", centroid[1])
    print("centroidy =", centroid[0])
    # Calculate central moments
    p0, p1, p2, p3 = 0, 1, 2, 3
    q0, q1, q2, q3 = 0, 1, 2, 3
    mqp1, mqp2, mqp3, mqp4, mqp5, mqp6, mqp7 = 0, 0, 0, 0, 0, 0, 0

    for y in range(0, int(h)):
        for x in range(0, int(w)):
            if grid[y][x] == 1:
                mqp1 += (float(x + 1) - centroid[1]) ** p2 * (float(y + 1) - centroid[0]) ** q0
                # M02
                mqp2 += (float(x + 1) - centroid[1]) ** p0 * (float(y + 1) - centroid[0]) ** q2
                # M11
                mqp3 += (float(x + 1) - centroid[1]) ** p1 * (float(y + 1) - centroid[0]) ** q1
                # M30
                mqp4 += (float(x + 1) - centroid[1]) ** p3 * (float(y + 1) - centroid[0]) ** q0
                # M12
                mqp5 += (float(x + 1) - centroid[1]) ** p1 * (float(y + 1) - centroid[0]) ** q2
                # M03
                mqp6 += (float(x + 1) - centroid[1]) ** p0 * (float(y + 1) - centroid[0]) ** q3
                # M21
                mqp7 += (float(x + 1) - centroid[1]) ** p2 * (float(y + 1) - centroid[0]) ** q1
    M20 = mqp1 / (m00 ** ((p2 + q0) / 2 + 1))
    M02 = mqp2 / (m00 ** ((p0 + q2) / 2 + 1))
    M11 = mqp3 / (m00 ** ((p1 + q1) / 2 + 1))
    M30 = mqp4 / (m00 ** ((p3 + q0) / 2 + 1))
    M03 = mqp5 / (m00 ** ((p0 + q3) / 2 + 1))
    M12 = mqp6 / (m00 ** ((p1 + q2) / 2 + 1))
    M21 = mqp7 / (m00 ** ((p2 + q1) / 2 + 1))
    S1 = M20 + M02
    S2 = (M20 + M02) * (M20 - M02) + 4 * M11 ** 2
    S3 = (M30 - 3 * M12) ** 2 + (M30 - 3 * M21) ** 2
    S4 = (M30 + M12) ** 2 + (M03 + M21) ** 2
    S5 = (M30 - 3 * M12) * (M30 + M12) * ((M30 + M12) ** 2 - 3 * (M03 + M21) ** 2) + (3 * M21 - M03) * (M03 + M21) * (3 * (M30 + M12) ** 2 - (M03 + M21) ** 2)
    S6 = (M20 - M02) * ((M30 + M12) ** 2 - (M03 + M21) ** 2) + 4 * M11 * (M30 + M12) * (M03 + M21)
    S7 = (3 * M21 - M03) * (M30 + M12) * ((M30 + M12) ** 2 - 3 * (M03 + M21) ** 2) + (M30 - 3 * M12) * (M21 + M02) * (3 * (M30 + M12) ** 2 - (M03 + M12) ** 2)

    print("S1 =", S1)
    print("S2 =", S2)
    print("S3 =", S3)
    print("S4 =", S4)
    print("S5 =", S5)
    print("S6 =", S6)
    print("S7 =", S7)
    return S1, S2, S3, S4, S5, S6, S7

def Template_matching(S1, S2):
    norm_distance = float(0)
    for i in range(0, 7):
        # Use 1-norm-distance
        norm_distance += float(abs(S1[i] - S2[i]))
    return norm_distance

# Pre-processing image input
print("\nPre-processing input")
# Image train
path_image = 'images/mangowithblackground.jpg'
image = cv2.imread(path_image, cv2.IMREAD_GRAYSCALE)
image_resize = cv2.resize(image,(90,90))
binnaryImage = convert_to_binary(image_resize)
grid, w, h = Image_to_grid(binnaryImage)
# Image detect 1
path_image_detect_1 = 'images/mangotest4.jpg'
image_detect_1 = cv2.imread(path_image_detect_1, cv2.IMREAD_GRAYSCALE)
image_resize_detect_1 = cv2.resize(image_detect_1,(90,90))
binnaryImage_detect_1 = convert_to_binary(image_resize_detect_1)
grid_detect_1, w_detect_1, h_detect_1 = Image_to_grid(binnaryImage_detect_1)
# Image detect 2
path_image_detect_2 = 'images/chillitest2.jpg'
image_detect_2 = cv2.imread(path_image_detect_2, cv2.IMREAD_GRAYSCALE)
image_resize_detect_2 = cv2.resize(image_detect_2,(90,90))
binnaryImage_detect_2 = convert_to_binary(image_resize_detect_2)
grid_detect_2, w_detect_2, h_detect_2 = Image_to_grid(binnaryImage_detect_2)

# Result pre-processing input
print("Image train")
for y in range(0, int(h)):
    for x in range(0, int(w)):
        print(grid[y][x], end=" ")
    print("")                
print('w = ',w, ' h = ', h)
print("Image detect 1")
for y in range(0, int(h_detect_1)):
    for x in range(0, int(w_detect_1)):
        print(grid_detect_1[y][x], end=" ")
    print("")        
print('w = ',w_detect_1, ' h = ', h_detect_1)
print("Image detect 2")
for y in range(0, int(h_detect_2)):
    for x in range(0, int(w_detect_2)):
        print(grid_detect_2[y][x], end=" ")
    print("")                
print('w = ',w_detect_2, ' h = ', h_detect_2)
# Hu'momnet
print("\nHu\'moment:")
print("S_train")
S_train = Humoment(grid, h, w)
print("\nS_detect_1")
S_detect_1 = Humoment(grid_detect_1, h_detect_1, w_detect_1)
print("\nS_detect_2")
S_detect_2 = Humoment(grid_detect_2, h_detect_2, w_detect_2)
# Use Template matching to implementation of classification
print("\nManhattan distance:")
norm_distance_1 = Template_matching(S_train, S_detect_1)
norm_distance_2 = Template_matching(S_train, S_detect_2)
# Result
print('norm distance for image detect 1: ',norm_distance_1)
print('norm distance for image detect 2: ',norm_distance_2)
# Classification
print("\nClassification:")
if norm_distance_2 > norm_distance_1 :
    print("First image detect has a mango and Second image detect has a chilli")
else:
    print("First image detect has a chilli and Second image detect has a mango")
print("")