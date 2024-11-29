# Đường dẫn tới tệp
file_path = "data_train\output_train\o0_test.ta"

# Đọc và xử lý nội dung tệp
with open(file_path, "r+") as file:
    lines = file.readlines()  # Đọc tất cả các dòng
    processed_lines = ["1\n" if line.strip() == "1" else "0\n" for line in lines]
    
    # Di chuyển con trỏ về đầu tệp và ghi lại dữ liệu đã xử lý
    file.seek(0)
    file.writelines(processed_lines)
    file.truncate()  # Xóa nội dung cũ còn dư nếu có

print(f"Tệp '{file_path}' đã được chỉnh sửa trực tiếp!")
