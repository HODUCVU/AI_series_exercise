import numpy as np
from PIL import Image

# Open image, convert to black and white mode
image = Image.open('binary_mangowithblackground.jpg').convert('1')
w, h = image.size

# Temporary NumPy array of type bool to work on
temp = np.array(image)

# Detect changes between neighbouring pixels
diff_y = np.diff(temp, axis=0)
diff_x = np.diff(temp, axis=1)

# Create union image of detected changes
temp = np.zeros_like(temp)
temp[:h-1, :] |= diff_y
temp[:, :w-1] |= diff_x

# Calculate distances between detected changes
diff_y = np.diff(np.nonzero(np.diff(np.sum(temp, axis=0))))
diff_x = np.diff(np.nonzero(np.diff(np.sum(temp, axis=1))))

# Calculate tile height and width
ht = np.median(diff_y[diff_y > 1]) + 2
wt = np.median(diff_x[diff_x > 1]) + 2

# Resize image w.r.t. tile height and width
# array = (~np.array(image.resize((int(w/wt), int(h/ht))))).astype(int)
array = (np.array(image.resize((int(w/wt), int(h/ht))))).astype(int)
for y in range(0, int(h/ht),1):
    for x in range(0,int(w/wt),1):
        print(array[y][x], end=" ")
    print("")
print(array)

'''

        # # Sử dụng hàm matchTemplate để so sánh cửa sổ trượt với mẫu
        # result = cv2.matchTemplate(test_window, window_patter, cv2.TM_CCOEFF_NORMED)

        # # Lấy vị trí của điểm tối ưu nhất trong kết quả
        # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        # print(max_val)
        # i = int(0)
        # # Nếu tìm thấy điểm tối ưu có độ tương đồng lớn hơn một ngưỡng nhất định
        # if max_val > 0.8:  # Sử dụng ngưỡng 0.6 làm ví dụ
        #     # Vẽ một hình chữ nhật quanh vùng tương đồng cao nhất
        #     top_left = max_loc
        #     bottom_right = (top_left[0] + w, top_left[1] + h)
        #     # cv2.rectangle(clone, top_left, bottom_right, (0, 255, 0), 2)
        #     cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #     cv2.imwrite("result%d.jpg" %i,clone)
        #     i += 1
        #     end = True
        # Hiển thị ảnh kết quả
        # cv2.imshow("Detected", clone)
'''

'''
def numpytoimage(numpy):
    numpy = numpy * 255
    image= Image.fromarray(numpy.astype(np.uint8))
    return image


reference = cv2.imread("ref.png",0)
_, thresh_ref = cv2.threshold(reference, 75, 255, 0)

extract = cv2.imread("extract.png",0)
_, thresh_extract = cv2.threshold(extract, 75, 255, 0)


C = np.zeros(shape=(len(thresh_ref), len(thresh_ref[0]), 3))

for i in range (0, thresh_ref.shape[0],1):
    for j in range(0, thresh_ref.shape[1], 1):
        if thresh_ref[i][j] == thresh_extract[i][j] and thresh_ref[i][j] == 0:
            C[i][j] = 1
        elif thresh_ref[i][j] == 0:
            C[i][j][0] = 0
            C[i][j][1] = 1
            C[i][j][2] = 0
        elif thresh_extract[i][j] == 0:
            C[i][j][0] = 1
            C[i][j][1] = 0
            C[i][j][2] = 0
        else:
            C[i][j][0] = 0.5
            C[i][j][1] = 0.5
            C[i][j][2] = 0.5

C_image = numpytoimage(C)
C_image.save("quality.png")

from PIL import Image

def from_img(imgfile, size, keep_ratio=True, reverse=False):
    def resample(img_, size):
        return img.resize(size, resample=Image.BILINEAR)            
    def makebw(img, threshold=200):
        edges = (255 if reverse else 0, 0 if reverse else 255)
        return img.convert('L').point(lambda x: edges[1] if x > threshold else edges[0], mode='1')
    img = Image.open(imgfile)
    if keep_ratio:
        ratio = max(size) / max(img.size)
        size = tuple(int(sz * ratio) for sz in img.size)
    return np.array(makebw(resample(img, size)), dtype=int)
'''