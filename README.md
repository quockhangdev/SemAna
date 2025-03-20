# SemAna

Là một dự án cơ bản về sử dụng mô hình Transformer-based cụ thể là Modern BERT để đánh giá tiêu cực hoặc tích cực về một bình luận dựa trên tập dữ liệu IMBD.

### Tạo môi trường ảo (Python 3.11.11 giống như Google Colab)

```
conda create -n semana python=3.11.11
```

### Cài đặt các thư viện cần thiết

```
conda install --yes --file requirements.txt
```

### Cấu hình của thiết bị lập trình

`Macbook Pro M1 Pro 16GB RAM`

Hệ điều hành: `Mac OS 15.3.2`

Do đó, khi cài đặt phần mềm sẽ có thể có một số thư viện/ công cụ không tương thích với Linux, Windows (WSL)...

### Chạy chương trình

Sau khi đã cài đầy đủ các thư viện/ công cụ cần thiết:

`bash inference.sh`

để chạy API sử dụng Litserve trên nền FastAPI./

## Lưu ý

Do dung lượng của mô hình khá lớn nên bạn cần tải mô hình từ Google Drive và copy-paste vào thư mục models có sẵn theo đúng cấu trúc.

https://drive.google.com/file/d/1s9EW311QLgEmXrYPD3pYLu8eh2pw5N9y/view?usp=drive_link

## Tác giả

`Trần Quốc Khang (twkan2001@gmail.com)`