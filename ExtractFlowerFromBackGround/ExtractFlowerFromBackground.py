import numpy as np
import matplotlib.pyplot as plt
import cv2

# Đọc ảnh
image = cv2.imread('BT1.png')
image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Chuyển sang không gian màu LAB
image_HSV = cv2.cvtColor(image_RGB, cv2.COLOR_RGB2HSV)
image_vectorized = image_HSV.reshape((-1, 3)).astype(np.float32)

# Số lượng cụm (clusters)
K = 3  # Một cụm cho bông hoa, hai cụm cho nền

# Khởi tạo các centroids ban đầu
centroids = image_vectorized[np.random.choice(image_vectorized.shape[0], K, replace=False), :]

# Khởi tạo biến lưu trữ cluster assignment
cluster_assignments = np.zeros((image_vectorized.shape[0], 1))

# Lặp lại việc cập nhật centroids và phân loại dữ liệu
for i in range(10):
    # Phân loại dữ liệu theo các centroids hiện tại
    for j in range(image_vectorized.shape[0]):
        distances = np.sum((centroids - image_vectorized[j, :])**2, axis=1)
        idx = np.argmin(distances)
        cluster_assignments[j] = idx

    # Cập nhật centroids bằng cách lấy trung bình của các điểm trong cùng một cluster
    for j in range(K):
        centroids[j, :] = np.mean(image_vectorized[cluster_assignments.ravel() == j, :], axis=0)

# Tính toán điểm số cho mỗi cụm
scores = []
for centroid in centroids:
    # Tính điểm dựa trên giá trị kênh a (màu vàng)
    a_score = centroid[1]  # Giả sử kênh a là thứ tự thứ 2 trong không gian màu LAB

    # Tính điểm dựa trên vị trí của cụm
    x, y = centroid[0], centroid[2]  # Giả sử kênh L và b là thứ tự thứ 1 và thứ 3
    image_center_x, image_center_y = image_HSV.shape[1] // 2, image_HSV.shape[0] // 2
    position_score = 1 / (np.sqrt((x - image_center_x)**2 + (y - image_center_y)**2) + 1)

    # Tổng điểm của cụm
    total_score = a_score + position_score
    scores.append(total_score)

# Xác định cụm nào là bông hoa dựa trên điểm số cao nhất
flower_centroid_idx = np.argmax(scores)
flower_mask = cluster_assignments.ravel() == flower_centroid_idx

# Áp dụng mặt nạ lên ảnh gốc, đặt nền thành màu đen
masked_image = np.copy(image_RGB)
masked_image[np.logical_not(flower_mask.reshape(image_RGB.shape[:2])), :] = 0

# Hiển thị ảnh đã loại bỏ nền
plt.subplot(1, 2, 1)
plt.imshow(image_RGB)
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(masked_image)
plt.title('Background Removed')
plt.axis('off')
plt.show()