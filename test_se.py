import data_train.library.module_DST as DST
import data_train.library.sentence as ST
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import time
# Đường dẫn file đầu vào, đầu ra và file log
file_input = 'data_train\\input_train\\content_question-test.ta'
file_output = 'data_train\\output_train\\o{}_test.ta'
file_log = 'data_train\\report\\test_{}.log'

Bt_test_T = []
# Đọc dữ liệu từ file_output cho từng chỉ số từ 0 đến 3
for i in range(6):
    with open(file_output.format(i), "r", encoding="utf-8") as output_file:
        # Đọc từng dòng trong file và lấy cột cho Bt_test_T
        lines = output_file.readlines()  # Đọc tất cả dòng trong file
        Bt_test_T.append([int(x) for x in [line.strip() for line in lines]])  # Lưu vào Bt_test_T
Bt_matrix = []
Bt_matrix_T = []
c=0
# Đọc dữ liệu từ file_input
start_time = time.time()
with open(file_input, "r", encoding="utf-8") as input_file:
    for input_line in input_file:
        c+=1
        dst_temp = DST.DST_block()  # Tạo một đối tượng DST_block
        dst_temp = ST.sentencess(input_line, dst_temp)  # Xử lý câu
        Bt_matrix.append(dst_temp.Bt)  # Lưu vào Bt_matrix
        print(c)
# Ghi nhận thời gian kết thúc
end_time = time.time()

# Tính thời gian đã trôi qua
elapsed_time = end_time - start_time
print(f"Thời gian thực thi: {elapsed_time:.4f} giây")

# Chuyển Bt_matrix thành Bt_matrix_T (ma trận chuyển vị)
Bt_matrix_T = [[row[i] for row in Bt_matrix] for i in range(len(Bt_matrix[0]))]
file_input_train=''
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
    if key == "output_train":
        output_train = value.strip("'")

# Lặp qua các chỉ số từ 0 đến 3 (tính cho từng ma trận)
for i in range(6):
    file_output_train=output_train.format(i)
    with open(file_output_train, "r") as file:
        numbers = file.readlines()
    # Chuyển các dòng từ chuỗi thành số nguyên và tìm số lớn nhất
    number_of_outputs = max(int(number.strip()) for number in numbers) + 1
    # Mở file log cho mỗi chỉ số
    with open(file_log.format(i), "w", encoding="utf-8") as log_file:
        # Tính confusion matrix
        print("in:{}".format(Bt_matrix_T[i]))
        print("out:{}".format(Bt_test_T[i]))
        
        

        cm = confusion_matrix(Bt_matrix_T[i], Bt_test_T[i], labels=[label for label in range(number_of_outputs)])
        
        # Tính các chỉ số đánh giás
        accuracy = accuracy_score(Bt_matrix_T[i], Bt_test_T[i],average='macro', zero_division=0,labels=[label for label in range(number_of_outputs)])
        precision = precision_score(Bt_matrix_T[i], Bt_test_T[i], average='macro', zero_division=0,labels=[label for label in range(number_of_outputs)])
        recall = recall_score(Bt_matrix_T[i], Bt_test_T[i], average='macro', zero_division=0,labels=[label for label in range(number_of_outputs)])
        f1 = f1_score(Bt_matrix_T[i], Bt_test_T[i], average='macro', zero_division=0,labels=[label for label in range(number_of_outputs)])
        
        # Ghi vào file log
        log_file.write(f"Model:{i}\n\n")
        log_file.write("Confusion Matrix:\n")
        log_file.write(f"{cm}\n\n")
        
        log_file.write(f"Accuracy: {accuracy:.4f}\n")
        log_file.write(f"Precision: {precision:.4f}\n")
        log_file.write(f"Recall: {recall:.4f}\n")
        log_file.write(f"F1 Score: {f1:.4f}\n")
