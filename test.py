import data_train.library.module_DST as DST
import data_train.library.sentence as ST

dst1 = DST.DST_block()
dst2 = DST.DST_block()
dst3 = DST.DST_block()

# Câu cần dự đoán
input1 = "What is comparator?"
input2 = "What is the second dominant pole in 2 satge?"
input3 = "What is the third dominant pole in 2 satge comparator?"


dst1 = ST.sentencess(input1,dst1)
print(":{}".format(dst1.Bt))
dst2 = ST.sentencess(input2,dst2)
print("user 2:{}".format(dst2.Bt))
dst3 = ST.sentencess(input3,dst3)
print("user 3:{}".format(dst3.Bt))





