import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.measure import regionprops, label

def initialize_centroids(points, k):
    # Chọn ngẫu nhiên các điểm khởi tạo làm centroids ban đầu
    indices = np.random.choice(points.shape[0], size=k, replace=False)
    return points[indices]

def closest_centroid(points, centroids):
    # Xác định centroid gần nhất cho từng điểm dữ liệu
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

def update_centroids(points, closest, centroids):
    # Cập nhật vị trí centroids dựa trên các điểm gần nhất
    return np.array([points[closest == k].mean(axis=0) for k in range(centroids.shape[0])])

def kmeans(points, k, max_iters=100):
    centroids = initialize_centroids(points, k)
    for _ in range(max_iters):
        closest = closest_centroid(points, centroids)
        new_centroids = update_centroids(points, closest, centroids)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return closest, centroids

def overlap(bbox1, bbox2):
    # Kiểm tra xem hai bounding box có giao nhau không
    return not (bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or 
                bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3])

def inside(bbox1, bbox2):
    return bbox1[0] >= bbox2[0] and bbox1[1] >= bbox2[1] and bbox1[2] <= bbox2[2] and bbox1[3] <= bbox2[3]

# Load ảnh và chuyển đổi thành numpy array
'data/Kmeans-1.png'
'data/ripe-fruit-2.png'
image = Image.open('data/Kmeans-1.png').convert('RGB')

# Kích thước ban đầu của ảnh
original_shape = image.size[::-1] + (3,) 

# Chuyển đổi ảnh thành mảng numpy các pixel
pixels = np.array(image).reshape(-1, 3)

# Số lượng clusters
k = 5
max_iters = 100

# Áp dụng thuật toán K-means để phân cụm các pixel
closest, centroids = kmeans(pixels, k, max_iters) 

# Chuyển đổi centroids sang kiểu dữ liệu integer
centroids = centroids.astype(int)

# Lấy chỉ số của cluster chứa màu sắc quả dâu
cluster_index = np.where(np.all(centroids == [181, 43, 19], axis=1))[0]

# Tạo ảnh đã phân đoạn dựa trên cluster được chỉ định
segmented_image = centroids[closest].reshape(image.size[1], image.size[0], 3)

# Tạo ảnh nhị phân
binary_image = np.zeros_like(segmented_image)
if cluster_index.size > 0:
    binary_image[closest.reshape(image.size[1], image.size[0]) == cluster_index[0]] = [181, 43, 19]

# Chuyển đổi ảnh nhị phân sang ảnh xám
binary_image_gray = np.dot(binary_image[..., :3], [0.2989, 0.5870, 0.1140])

# Label các vùng trong ảnh nhị phân
labels = label(binary_image_gray)

# Tính toán các thuộc tính của các vùng
regions = regionprops(labels)

# Nhóm các bounding box dựa trên sự chồng chéo
bbox_groups = []
for region in regions:
    bbox = region.bbox
    for group in bbox_groups:
        if any(overlap(bbox, other_bbox) for other_bbox in group):
            group.append(bbox)
            break
    else: 
        bbox_groups.append([bbox]) 

# Mở rộng bounding boxes đã nhóm
expanded_bboxes = [(min(bbox[0] for bbox in group), min(bbox[1] for bbox in group),
                    max(bbox[2] for bbox in group), max(bbox[3] for bbox in group))
                   for group in bbox_groups]

# Lọc các bounding boxes dựa trên kích thước và sự chồng chéo
filtered_bboxes = [bbox for bbox in expanded_bboxes if not any(
    inside(bbox, other_bbox) for other_bbox in expanded_bboxes if bbox != other_bbox) and
                   (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) >= 1200]

# Hiển thị ảnh gốc và bounding boxes đã lọc
fig, ax = plt.subplots(1)
ax.imshow(image)
for bbox in filtered_bboxes:
    minr, minc, maxr, maxc = bbox
    rect = Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, color='yellow')
    ax.add_patch(rect)
plt.show()
