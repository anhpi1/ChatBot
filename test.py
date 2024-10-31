import data_train.library.module_DST as DST
import data_train.library.sentence as ST

dst = DST.DST_block()

# Câu cần dự đoán
input1 = "i love my cat, my cat is so dump and i like this"
input2 = "i love my dog, my dog is so dump and i like this"
input3 = "i love my frog, my frog is so dump and i like this"


dst = ST.sentencess(input1,dst)
dst = ST.sentencess(input2,dst)
dst = ST.sentencess(input3,dst)
print(dst)



