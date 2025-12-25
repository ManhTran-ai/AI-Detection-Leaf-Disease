"""
Utility script để đọc và trích xuất text từ file PDF.
Có thể sử dụng để import PDF vào workspace và agent có thể hiểu được nội dung.
"""

import os
from pathlib import Path
from typing import Optional, List
import pdfplumber


def extract_text_from_pdf(pdf_path: str, output_path: Optional[str] = None) -> str:
    """
    Trích xuất toàn bộ text từ file PDF.
    
    Args:
        pdf_path: Đường dẫn đến file PDF
        output_path: (Optional) Đường dẫn để lưu text ra file .txt
    
    Returns:
        String chứa toàn bộ text từ PDF
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File PDF không tồn tại: {pdf_path}")
    
    text_content = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            print(f"Đang đọc PDF: {pdf_path}")
            print(f"Số trang: {len(pdf.pages)}")
            
            for i, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text:
                    text_content.append(f"\n--- Trang {i} ---\n")
                    text_content.append(text)
                    print(f"Đã đọc trang {i}/{len(pdf.pages)}")
            
        full_text = "\n".join(text_content)
        
        # Lưu ra file text nếu có output_path
        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(full_text)
            print(f"\nĐã lưu text vào: {output_path}")
        
        return full_text
    
    except Exception as e:
        raise Exception(f"Lỗi khi đọc PDF: {str(e)}")


def extract_text_by_page(pdf_path: str) -> List[str]:
    """
    Trích xuất text từng trang riêng biệt.
    
    Args:
        pdf_path: Đường dẫn đến file PDF
    
    Returns:
        List các string, mỗi string là text của một trang
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File PDF không tồn tại: {pdf_path}")
    
    pages_text = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                pages_text.append(text if text else "")
        
        return pages_text
    
    except Exception as e:
        raise Exception(f"Lỗi khi đọc PDF: {str(e)}")


def get_pdf_info(pdf_path: str) -> dict:
    """
    Lấy thông tin cơ bản về PDF.
    
    Args:
        pdf_path: Đường dẫn đến file PDF
    
    Returns:
        Dictionary chứa thông tin PDF
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File PDF không tồn tại: {pdf_path}")
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            info = {
                "file_path": pdf_path,
                "num_pages": len(pdf.pages),
                "metadata": pdf.metadata if hasattr(pdf, 'metadata') else {}
            }
        return info
    
    except Exception as e:
        raise Exception(f"Lỗi khi đọc thông tin PDF: {str(e)}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Cách sử dụng:")
        print("  python src/utils/pdf_reader.py <path_to_pdf> [output_txt_path]")
        print("\nVí dụ:")
        print("  python src/utils/pdf_reader.py docs/document.pdf")
        print("  python src/utils/pdf_reader.py docs/document.pdf docs/document.txt")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        # Hiển thị thông tin PDF
        info = get_pdf_info(pdf_path)
        print(f"\n📄 Thông tin PDF:")
        print(f"  - File: {info['file_path']}")
        print(f"  - Số trang: {info['num_pages']}")
        if info['metadata']:
            print(f"  - Metadata: {info['metadata']}")
        
        # Trích xuất text
        text = extract_text_from_pdf(pdf_path, output_path)
        print(f"\n✅ Đã trích xuất {len(text)} ký tự từ PDF")
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        sys.exit(1)










