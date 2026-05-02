import cv2
import os
import csv
import numpy as np

def denoise(img_path):
    #OpenCV đọc file
    img = cv2.imread(img_path)
    if img is None:
        return None
    #Chuyển ảnh màu sang ảnh xám cvtColor = convert color , cv2.COLOR_BGR2GRAY = mã chuển màu
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Giai đoạn tách nền
    #THRESH_OTSU thuật toán tự đồng tìm ngưỡng tách nền, THRESH_BINARY tô trắng đen dựa theo ngưỡng
    thresh_hold, mask = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    #Giai đoạn khử nhiễu cho ảnh
    #Khai báo kernel 5x5 để duyện ảnh, np.uint8 -> khai báo kiểu dữ liệu
    kernel = np.ones((5, 5), np.uint8)
    #Opening: Diệt các hạt nhiễu trắng li ti bên ngoài hoa + bào viền hoa một tí.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #Closing: Diệt các hạt nhiễu đen li ti trong hoa + lấp đầy phần đã bào
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    ##Đặc trưng không cần đo kích thước nên không cần bước phân vùng bao và định danh
    return mask

def feture_extract(mask):
    # mask lỗi
    if mask is None:
        return None
    # Tìm đường bao ngoài cùng = bao toàn bộ vật thể(cây hoa)
    #RETR_EXTERNAL: chế độ tìm đường bao -> retrive ngoài cùng, CHAIN_APPROX_SIMPLE -> nén dữ liệu chỉ lưu lại các điểm tọa độ 2 đầu các đoạn thẳng của đường bao
    contours, tree = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Ảnh đen thui
    if(len(contours) == 0):
        return None
    # Lấy vật thể lớn nhất, vì có thể sót lại một vài pixel nhiễu trắng chúng vần đc trả về đường bao
    max_contour = max(contours, key=cv2.contourArea)
    # Lấy diện tích vật thể
    flower_area = cv2.contourArea(max_contour)

    # Diện tích bé hơn 100 bỏ
    if(flower_area < 100):
        return None

    # Giai đoạn trích xuất đặc trưng
    # Đặc trưng 1 Aspect ratio
    rect = cv2.minAreaRect(max_contour)
    (x, y), (width, height), angle = rect

    # Tránh lỗi chia cho 0 nếu ảnh bị lỗi nét mỏng
    if min(width, height) == 0: return None

    # Lấy cạnh dài chia cạnh ngắn
    aspect_ratio = max(width, height) / min(width, height)

    # Đặc trưng 2 Solidity
    hull = cv2.convexHull(max_contour)
    hull_area = cv2.contourArea(hull)
    solidity = float (flower_area) / hull_area if hull_area > 0 else 0

    # Đặc trưng 3: moment hu1 và hu2
    momments  = cv2.moments(max_contour)
    hu_moments = cv2.HuMoments(momments)

    # Dùng Logarit để ép số liệu Hu (bản chất cực kỳ nhỏ) về dạng dễ đọc
    h1 = hu_moments[0][0]
    h2 = hu_moments[1][0]

    h1_log = -np.sign(h1) * np.log10(abs(h1)) if h1 != 0 else 0
    h2_log = -np.sign(h2) * np.log10(abs(h2)) if h2 != 0 else 0

    # TRẢ VỀ KẾT QUẢ
    # Làm tròn 4 chữ số thập phân cho đẹp file CSV
    return [round(aspect_ratio, 4), round(solidity, 4), round(h1_log, 4), round(h2_log, 4)]

# 1. Tạo file CSV để lưu đặc trưng
with open('flower_feature.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Ten_File", "Loai_Hoa", "Solidity", "Aspect_Ratio", "Hu_h1", "Hu_h2"])

    # 2. Quét qua từng thư mục ảnh
    data_dir = 'Dataset_Hoa'
    for flower in os.listdir(data_dir):
        flower_class_path = os.path.join(data_dir, flower)

        # 3. Đọc từng bức ảnh
        for flower_image in os.listdir(flower_class_path):
            img_path = os.path.join(flower_class_path, flower_image)
           #Xử lý ảnh đủ để trích xuất các đặc trưng cần thiết: Aspect Ratio, Solidity, Hu1, Hu2
           #Aspect Ratio = cạnh dài / ngắn => Biết vật thể dài hay tròn
           #Solidity = area vật thể / bao lồi (convex hull) => dựa trên giá trị từ 0 -> 1 biết hình dạng vật thể sao, tròn...
           #momen hu1: Đo độ phân tán của các pixel quanh vật thể, bông hoa càng bung thì h1 càng lớn
           #momen hu2: Đo độ thon dài của vật thể, vd như bị gió thổi
            mask = denoise(img_path)
            features = feture_extract(mask)
            if features is not None:
                # Ghi 1 dòng vào file CSV gồm: Tên ảnh, Loại hoa, và 4 thông số
                dong_du_lieu = [flower_image, flower] + features
                writer.writerow(dong_du_lieu)
                print(f"✅ Đã xử lý: {flower_image}")
            else:
                print(f"❌ Bỏ qua (Lỗi ảnh hoặc không có hoa): {flower_image}")
print("Đã trích xuất xong đặc trưng và lưu vào CSV!")

