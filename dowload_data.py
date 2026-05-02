from bing_image_downloader import downloader

# Danh sách 3 loài hoa bạn muốn tải
danh_sach_hoa = ["Setosa flower", "Versicolor flower", "Virginica flower"]

for loai_hoa in danh_sach_hoa:
    print(f"Đang tải ảnh cho loài: {loai_hoa}...")

    downloader.download(
        loai_hoa,
        limit=35,  # Số lượng ảnh muốn tải mỗi loài (35 x 3 = 105 ảnh)
        output_dir='Dataset_Hoa',  # Tên thư mục tổng
        adult_filter_off=True,  # Tắt lọc nội dung người lớn để tránh sót ảnh
        force_replace=False,  # Không tải đè nếu ảnh đã có
        timeout=60,  # Chờ tối đa 60s cho mỗi ảnh
        verbose=True  # In quá trình tải ra màn hình
    )

print("Đã tải xong toàn bộ dữ liệu!")