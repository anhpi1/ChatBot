import pyodbc

# Thông tin kết nối SQL Server
server = 'DESKTOP-1MU0IU3\SQLEXPRESS'
database = 'comparator'
username = ''
password = ''

# Kết nối đến SQL Server
conn = pyodbc.connect(
    f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
)

# Tạo một đối tượng cursor
cursor = conn.cursor()


# Thực thi truy vấn để lấy dữ liệu
cursor.execute("SELECT content FROM dbo.question;")
rows = cursor.fetchall()

# Ghi dữ liệu vào file x.txt
with open("data_train\input_train\content_question.ta", "w", encoding="utf-8") as file:
    for row in rows:
        file.write(f"{row.content}\n")

# Đóng kết nối
cursor.close()
conn.close()
