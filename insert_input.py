import pyodbc

# Thông tin kết nối SQL Server
server = ''
database = ''
username = ''
password = ''
content_question = ''
command_sever_get_input = ''

# tải tham số
with open("parameter.ta", "r") as file:
    lines = file.readlines()
for line in lines:
    # Bỏ qua các dòng trống
    if not line.strip():
        continue
    # Tách dòng thành key và value
    key, value = line.split(" = ")
    key = key.strip()
    value = value.strip()
    if key == "server":
        server = value.strip("'")
    if key == "database":
        database = value.strip("'")
    if key == "username":
        username = value.strip("'")
    if key == "file_input_train":
        password = value.strip("'")
    if key == "content_question":
        content_question = value.strip("'")
    if key == "command_sever_get_input":
        command_sever_get_input = value.strip("'")

# Kết nối đến SQL Server
conn = pyodbc.connect(
    f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
)

# Tạo một đối tượng cursor
cursor = conn.cursor()


# Thực thi truy vấn để lấy dữ liệu
cursor.execute(command_sever_get_input)
rows = cursor.fetchall()

# Ghi dữ liệu vào file content_question.txt
with open(content_question, "w", encoding="utf-8") as file:
    for row in rows:
        file.write(f"{row.content}\n")

# Đóng kết nối
cursor.close()
conn.close()