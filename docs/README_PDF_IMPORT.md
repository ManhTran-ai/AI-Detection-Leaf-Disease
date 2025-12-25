# 📄 Hướng Dẫn Import PDF vào Workspace

## Cách 1: Copy file PDF trực tiếp

1. Copy file PDF của bạn vào thư mục `docs/` hoặc bất kỳ thư mục nào trong workspace
2. Agent có thể đọc file PDF thông qua script `src/utils/pdf_reader.py`

## Cách 2: Sử dụng script Python để trích xuất text

### Bước 1: Cài đặt thư viện

```bash
pip install -r requirements.txt
```

Script sẽ tự động cài `PyPDF2` và `pdfplumber`.

### Bước 2: Chạy script để trích xuất text

```bash
# Chỉ hiển thị text trong terminal
python src/utils/pdf_reader.py docs/your_file.pdf

# Trích xuất và lưu ra file .txt
python src/utils/pdf_reader.py docs/your_file.pdf docs/your_file.txt
```

### Bước 3: Agent có thể đọc file .txt

Sau khi trích xuất, agent có thể đọc file `.txt` để hiểu nội dung PDF.

## Sử dụng trong code Python

```python
from src.utils.pdf_reader import extract_text_from_pdf, get_pdf_info

# Lấy thông tin PDF
info = get_pdf_info("docs/document.pdf")
print(f"Số trang: {info['num_pages']}")

# Trích xuất toàn bộ text
text = extract_text_from_pdf("docs/document.pdf")
print(text)

# Lưu ra file text
extract_text_from_pdf("docs/document.pdf", "docs/document.txt")
```

## Lưu ý

- PDF có thể chứa hình ảnh, bảng biểu phức tạp → text có thể không hoàn hảo
- Nếu PDF được scan (hình ảnh), cần OCR (Optical Character Recognition) để đọc
- File PDF lớn có thể mất thời gian để xử lý










