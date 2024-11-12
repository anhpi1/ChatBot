import pyodbc
import ast
# Thông tin kết nối SQL Server
server = ''
database = ''
username = ''
password = ''
output_train = ''
command_sever_get_output_train = ''
command_connect_sever = ''
topics = []
number_of_model=0

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
    if key == "password":
        password = value.strip("'")
    if key == "output_train":
        output_train = value.strip("'")
    if key == "command_sever_get_output_train":
        command_sever_get_output_train = value.strip("'")
    if key == "command_connect_sever":
        command_connect_sever = value.strip("'")
    if key == "number_of_model" and value.isdigit():
        number_of_model = int(value)
    if line.strip().startswith("topics = "):
        # Trích xuất chuỗi sau 'topics = '
        topics_str = line.strip()[len("topics = "):].strip()
            
        # Dùng ast.literal_eval để chuyển chuỗi thành danh sách Python
        topics = ast.literal_eval(topics_str)



# Kết nối đến SQL Server
conn = pyodbc.connect(command_connect_sever.format(server,database,username,password)    
)



# Tạo một đối tượng cursor
cursor = conn.cursor()

for i in range(0,number_of_model):
    # Thực thi truy vấn với tên cột từ biến
    query = command_sever_get_output_train.format(topics[i])
    cursor.execute(query)
    rows = cursor.fetchall()

    # Ghi dữ liệu vào file output_train
    with open(output_train.format(i), "w", encoding="utf-8") as file:
        for row in rows:
            # Thay None thành 0 trong mỗi dòng
            row_data = [0 if item is None else item for item in row]
            # Nối các phần tử và ghi vào file
            file.write(" ".join(str(item) for item in row_data) + "\n")

# Đóng kết nối
cursor.close()
conn.close()
