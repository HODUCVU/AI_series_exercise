import os, sys
import cv2

def convert_to_binary(img_grayscale, thresh=100):
    thresh, img_binary = cv2.threshold(img_grayscale, thresh, maxval=255, type=cv2.THRESH_BINARY)
    return img_binary

if __name__ == "__main__":
    # assert len(sys.argv) == 2, '[USAGE] $ python3 color2binary.py mangowithblackground.jpg'
    # input_image_path = sys.argv[1]
    
    # assert os.path.isfile(input_image_path), 'Image not found @ %s' % input_image_path
    input_image_path = "mangotest.jpg"
    # read color image with grayscale flag: "cv2.IMREAD_GRAYSCALE"
    img_grayscale = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)       # shape: (960, 960)
    # print grayscale image
    # cv2.imwrite('grey_%s' % input_image_path, img_grayscale)
    # print('Saved grayscale image @ grey_%s' % input_image_path)
    
    img_binary = convert_to_binary(img_grayscale, thresh=100)
    cv2.imwrite('binary_%s' % input_image_path, img_binary)
    print('Saved binary image @ binary_%s' % input_image_path)
# image = cv2.imread("binary_mangowithblackground.jpg", cv2.IMREAD_GRAYSCALE)
