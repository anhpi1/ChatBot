import pandas as pd
import pyodbc

# Kết nối đến SQL Server
conn = pyodbc.connect(
    'DRIVER={SQL Server};'
    'SERVER=DESKTOP-1MU0IU3\SQLEXPRESS;'  # Tên máy chủ SQL Server của bạn
    'DATABASE=comparator;'  # Tên cơ sở dữ liệu
    'Trusted_Connection=yes;'
)

# Viết truy vấn SQL
query = "select * from question"  # Thay 'ten_bang' bằng tên bảng của bạn

# Đọc dữ liệu từ SQL Server vào DataFrame
df = pd.read_sql(query, conn)

# Lưu DataFrame vào file Excel
df.to_excel('data_train/thong_ke_tan_suat_cau_tra_loi.xlsx', index=False)

# Đóng kết nối
conn.close()

print("Dữ liệu đã được lưu vào file du_lieu.xlsx")
