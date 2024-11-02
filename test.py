import data_train.library.module_DST as DST
import data_train.library.sentence as ST

dst1 = DST.DST_block()
dst2 = DST.DST_block()
dst3 = DST.DST_block()

# Câu cần dự đoán
input1 = "Why would you use two stages instead of only one in 2 stage comparator?"
input2 = "In DC normal operation, What is the input resistance of 2 stage ?"
input3 = "What is the Miller compensation?"


dst1 = ST.sentencess(input1,dst1)
print("user 1:{}".format(dst1.Bt))
dst2 = ST.sentencess(input2,dst2)
print("user 2:{}".format(dst2.Bt))
dst3 = ST.sentencess(input3,dst3)
print("user 3:{}".format(dst3.Bt))





