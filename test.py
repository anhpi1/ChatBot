import data_train.library.module_DST as DST
import data_train.library.sentence as ST
import data_train.library.printf as printf
import data_train.library.train_TNN as TNN
# dst1 = DST.DST_block()
# dst2 = DST.DST_block()
# dst3 = DST.DST_block()

# # Câu cần dự đoán
input1 = "What is comparator?"
# input2 = "What is the second dominant pole in 2 satge?"
# input3 = "What is the third dominant pole in 2 satge comparator?"


# dst1 = ST.sentencess(input1,dst1)
# print("user 1:{}".format(dst1.Bt))
# printf.print_Bt(dst1.Bt)

# dst2 = ST.sentencess(input2,dst2)
# print("user 2:{}".format(dst2.Bt))
# printf.print_Bt(dst2.Bt)

# dst3 = ST.sentencess(input3,dst3)
# print("user 3:{}".format(dst3.Bt))
# printf.print_Bt(dst3.Bt)

TNN.update_weights_models(0 , input1, 3)
# What is the dominant pole of a Miller compensated two stage comparator?
# [what]
# [2 stage comparator]:1
# [NULL]
# [Phase margin]1
# [NULL]
# [Miller compensation components]
# [NULL]

# [what]
# [NULL]
# [Input stager]
# [Phase margin]
# [NULL]
# [Miller compensation components]
# [NULL]






