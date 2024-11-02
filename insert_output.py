import pyodbc

# Thông tin kết nối SQL Server
server = 'DESKTOP-1MU0IU3\SQLEXPRESS'
database = 'comparator'
username = ''
password = ''

# Tên cột được lưu trong biến
name=['topology_id','structure_id','performance_metric_id','applications_id','components_id','tools_id']

# Kết nối đến SQL Server
conn = pyodbc.connect(
    f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
)

# Tạo một đối tượng cursor
cursor = conn.cursor()

for i in range(0,6):
    # Thực thi truy vấn với tên cột từ biến
    query = f"SELECT {name[i]} FROM dbo.question;"
    cursor.execute(query)
    rows = cursor.fetchall()

    # Ghi dữ liệu vào file x.txt
    with open("data_train\output_train\o{}.ta".format(i+1), "w", encoding="utf-8") as file:
        for row in rows:
            # Thay None thành 0 trong mỗi dòng
            row_data = [0 if item is None else item for item in row]
            # Nối các phần tử và ghi vào file
            file.write(" ".join(str(item) for item in row_data) + "\n")

# Đóng kết nối
cursor.close()
conn.close()
