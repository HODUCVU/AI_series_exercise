import cv2
import numpy as np
from PIL import Image

def Image_to_grid(window, test, miss=0):
    # Open image, convert to black and white mode
    image = Image.fromarray(window).convert('1')
    w, h = image.size
    
    imagetest = Image.fromarray(test).convert('1')

    # Temporary NumPy array of type bool to work on
    temp = np.array(image, dtype=bool)
    temptest = np.array(imagetest, dtype=bool)

    # Detect changes between neighboring pixels
    diff_y = np.abs(np.diff(temp, axis=0).astype(bool))  # Convert diff_y to bool
    diff_x = np.abs(np.diff(temp, axis=1).astype(bool))  # Convert diff_x to bool
    
    diff_ytest = np.abs(np.diff(temptest, axis=0).astype(bool))  
    diff_xtest = np.abs(np.diff(temptest, axis=1).astype(bool))  
    
    # Create a union image of detected changes
    temp = np.zeros_like(temp, dtype=bool)
    temp[:h-1, :] |= diff_y
    temp[:, :w-1] |= diff_x
    
    temptest = np.zeros_like(temptest, dtype=bool)
    temptest[:h-1, :] |= diff_ytest
    temptest[:, :w-1] |= diff_xtest
    
    # Calculate distances between detected changes
    diff_y = np.diff(np.nonzero(np.diff(np.sum(temp, axis=0))))
    diff_x = np.diff(np.nonzero(np.diff(np.sum(temp, axis=1))))

    diff_ytest = np.diff(np.nonzero(np.diff(np.sum(temptest, axis=0))))
    diff_xtest = np.diff(np.nonzero(np.diff(np.sum(temptest, axis=1))))
    
    # Calculate tile height and width
    ht = np.median(diff_y[diff_y > 1]) + 2
    wt = np.median(diff_x[diff_x > 1]) + 2
    
    httest = np.median(diff_ytest[diff_ytest > 1]) + 2
    wttest = np.median(diff_xtest[diff_xtest > 1]) + 2
    
    # Resize image w.r.t. tile height and width
    if (ht > 0 and wt > 0) and (httest > 0 and wttest > 0):  # Ensure that ht/httest and wt/wttest are greater than 0
        array = np.array(image.resize((int(w/wt), int(h/ht)))).astype(int)
        arraytest = np.array(imagetest.resize((int(w/wt), int(h/ht)))).astype(int)
        for y in range(0, int(h/ht)):
            for x in range(0, int(w/wt)):
                print(array[y][x], end=" ")
                
                if array[y][x] != arraytest[y][x]:
                    miss += 1
            print(" | ",end="")
            for x in range(0, int(w/wt)):
                print(arraytest[y][x], end=" ")
            print("")
        return miss
    else:
        print("Invalid tile height or width.")
        return None
    

def convert_to_binary(img_grayscale, thresh=100):
    thresh, img_binary = cv2.threshold(img_grayscale, thresh, maxval=255, type=cv2.THRESH_BINARY)
    return img_binary

def sliding_window(image, step_size, window_size):
    # get the window and image sizes
    h, w = window_size
    image_h, image_w = image.shape[:2]
    # loop over the image, taking steps of size `step_size`
    for y in range(0, image_h, step_size):
        for x in range(0, image_w, step_size):
            # define the window
            window = image[y:y + h, x:x + w]
            # if the window is below the minimum window size, ignore it
            if window.shape[:2] != window_size:
                continue
            # yield the current window
            yield (x, y, window)
            

path_image_test = "mangotest.jpg"
path_image_patter = "mangowithblackground.jpg"

image_patter = cv2.imread(path_image_patter, cv2.IMREAD_GRAYSCALE)
window_patter = cv2.resize(image_patter,(90,90))
window_patter = convert_to_binary(window_patter)

image_test = cv2.imread(path_image_test)
h_o, w_o, _ = image_test.shape # to resize clone to origin size
image_test_gray = cv2.imread(path_image_test, cv2.IMREAD_GRAYSCALE)
image_test_binnary = convert_to_binary(image_test_gray)
# Size of window
w, h = 90,90
# miss point
miss = np.zeros(1000)
countmiss = 0
isObject = False

while True:
    
    for (x, y, window) in sliding_window(image_test, step_size=30, window_size=(w, h)):

        test_window = image_test_binnary[y:y+h, x:x+w]
        miss[countmiss] = Image_to_grid(window_patter, test_window)
        
        print("Miss: ", miss[countmiss])
        
        clone = image_test.copy()
        cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        if miss[countmiss] is not None and miss[countmiss] < 80:
            none_zero = miss[miss!=0]
            if len(none_zero) > 0 and miss[countmiss] <= np.min(none_zero):
                clone = cv2.resize(clone, (w_o, h_o))
                cv2.imwrite('bounding_box_%d_%s' %(miss[countmiss], path_image_test) , clone)
                isObject = True
            
        cv2.imshow("Window", clone)
        cv2.imshow("Window_test", test_window)
        cv2.imshow("Window_patter", window_patter)

        cv2.waitKey(10)
        print("")
    if image_test.shape[0] <= 1 or image_test.shape[1] <= 1:
        break
    image_test = cv2.pyrDown(image_test)
    image_test_gray = cv2.cvtColor(image_test, cv2.COLOR_BGR2GRAY)
    image_test_binnary = convert_to_binary(image_test_gray)
if not isObject:
    print("There are no mangoes in the photo")
    
