import data_train.library.module_DST as DST
import data_train.library.sentence as ST

dst = DST.DST_block()

# Câu cần dự đoán
input1 = "i love my cat, my cat is so dump and i like this"
input2 = "Kangaroos are known for their powerful hind legs."
input3 = "Birds are known for their colorful feathers."

dst = ST.sentencess(input1,dst)

dst = ST.sentencess(input2,dst)

dst = ST.sentencess(input3,dst)

print(dst)





