import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn import tree
import seaborn as sns

def train_model():
    print("1. Đang đọc dữ liệu từ file CSV...")
    # Dùng đặc trưng mẫu trên mạng
    df = pd.read_csv('IRIS.csv')

    # Kiểm tra xem file có dữ liệu không
    if df.empty:
        print("Lỗi: File CSV trống. Vui lòng kiểm tra lại bước trích xuất ảnh!")
        return

    # Vẽ biểu đồ phân tán chiều dài, rộng
    sns.scatterplot(data=df, x='petal_length', y='petal_width', hue='species', s=100)
    plt.title("Sự phân tách hoàn hảo của 3 loài Iris dựa trên Cánh hoa")
    plt.show()

    # 2. Phân chia Dữ liệu: Đặc trưng (X) và Nhãn (y)
    # Lấy 4 cột số liệu làm Đặc trưng để học. Cột 'Ten_File' không có tác dụng phân loại nên ta bỏ qua.
    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    # Cột chứa tên loại hoa chính là Nhãn (Kết quả máy cần đoán)
    y = df['species']

    print(f"-> Tìm thấy {len(df)} mẫu dữ liệu.")

    # 3. Chia tập Dữ liệu (Train/Test Split)
    # Chia 80% để học (Train), 20% để làm bài kiểm tra (Test)
    # random_state=42 giúp kết quả xáo trộn luôn giống nhau ở mọi lần chạy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"-> Đã chia: {len(X_train)} mẫu để huấn luyện, {len(X_test)} mẫu để kiểm tra.")

    # 4. Khởi tạo và Huấn luyện Cây Quyết Định (Decision Tree)
    print("\n2. Đang huấn luyện Cây Quyết Định...")
    # Dùng chỉ số 'gini' để đo độ thuần khiết
    # max_depth=5: Chỉ cho phép cây mọc tối đa 5 tầng để chống học vẹt (Overfitting)
    model = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)

    # Nạp tập dữ liệu vào model để train
    model.fit(X_train, y_train)

    # 5. Đánh giá Mô hình (Làm bài kiểm tra)
    print("\n3. Đánh giá kết quả trên tập Kiểm tra (Test):")
    # Lấy tập đề thi (X_test) cho AI đoán thử
    y_pred = model.predict(X_test)

    # Chấm điểm
    accuracy = accuracy_score(y_test, y_pred)
    print(f"-> Độ chính xác tổng thể (Accuracy): {accuracy * 100:.2f}%")

    print("\nBáo cáo chi tiết từng loại hoa:")
    print(classification_report(y_test, y_pred))

    # 6. TRỰC QUAN HÓA CÂY QUYẾT ĐỊNH (Phần thú vị nhất!)
    print("4. Đang vẽ biểu đồ Cây Quyết Định...")
    plt.figure(figsize=(50, 25), dpi=300)  # Chỉnh kích thước khung hình
    tree.plot_tree(model,
                   feature_names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
                   class_names=model.classes_,
                   filled=True,
                   rounded=True,
                   fontsize=11)
    plt.title("Biểu đồ Phân nhánh của Cây Quyết Định")
    plt.show()


# Chạy chương trình
if __name__ == "__main__":
    train_model()