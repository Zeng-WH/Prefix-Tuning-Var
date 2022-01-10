# 将数据集的格式转化成对抗学习的格式

import numpy as np

with open("/home/hadoop-aipnlp/cephfs/data/zengweihao02/NAACL2021/Prefix-Tuning/seq2seq/data/NAACL2021_Data/train_as_test/train.source", "r") as r1:
    train_source = r1.readlines()
with open("/home/hadoop-aipnlp/cephfs/data/zengweihao02/NAACL2021/Prefix-Tuning/seq2seq/data/NAACL2021_Data/train_as_test/train.target", "r") as r1:
    train_target = r1.readlines()

with open("/home/hadoop-aipnlp/cephfs/data/zengweihao02/NAACL2021/Prefix-Tuning/seq2seq/data/NAACL2021_Data/train_as_test/train.oracle", "r") as r1:
    train_oracle = r1.readlines()



with open("/home/hadoop-aipnlp/cephfs/data/zengweihao02/NAACL2021/Prefix-Tuning/seq2seq/data/NAACL2021_Data/train_as_test/val.source", "r") as r1:
    val_source = r1.readlines()
with open("/home/hadoop-aipnlp/cephfs/data/zengweihao02/NAACL2021/Prefix-Tuning/seq2seq/data/NAACL2021_Data/train_as_test/val.target", "r") as r1:
    val_target = r1.readlines()

with open("/home/hadoop-aipnlp/cephfs/data/zengweihao02/NAACL2021/Prefix-Tuning/seq2seq/data/NAACL2021_Data/train_as_test/val.oracle", "r") as r1:
    val_oracle = r1.readlines()



with open("/home/hadoop-aipnlp/cephfs/data/zengweihao02/NAACL2021/Prefix-Tuning/seq2seq/data/NAACL2021_Data/train_as_test/test.source", "r") as r1:
    test_source = r1.readlines()
with open("/home/hadoop-aipnlp/cephfs/data/zengweihao02/NAACL2021/Prefix-Tuning/seq2seq/data/NAACL2021_Data/train_as_test/test.target", "r") as r1:
    test_target = r1.readlines()

with open("/home/hadoop-aipnlp/cephfs/data/zengweihao02/NAACL2021/Prefix-Tuning/seq2seq/data/NAACL2021_Data/train_as_test/test.oracle", "r") as r1:
    test_oracle = r1.readlines()


#train_source = train_source[0:len(test_source)]
#train_target = train_target[0:len(test_target)]
#train_oracle = train_oracle[0:len(test_oracle)]

train_domain_label = [int(1) for x in train_source]

#train_id = np.arrange(len(train_source))


val_source = val_source

val_target = val_target

val_oracle = val_oracle
val_domain_label = [int(1) for x in val_source]

#test_source = test_source
test_domain_label = [int(0) for x in test_source]



train_source.extend(test_source)
train_target.extend(test_target)

train_oracle.extend(test_oracle)

train_domain_label.extend(test_domain_label)

train_id = np.arange(len(train_source))

np.random.shuffle(train_id)

train_source_new = [train_source[id] for id in train_id]
train_target_new = [train_target[id] for id in train_id]
train_oracle_new = [train_oracle[id] for id in train_id]
train_domain_label_new = [train_domain_label[id] for id in train_id]

with open('/home/hadoop-aipnlp/cephfs/data/zengweihao02/NAACL2021/Prefix-Tuning/seq2seq_adversal/data/train_adversial/train_as_test/train.source', "w") as w:
    for item in train_source_new:
        w.write(item)

with open('/home/hadoop-aipnlp/cephfs/data/zengweihao02/NAACL2021/Prefix-Tuning/seq2seq_adversal/data/train_adversial/train_as_test/train.target', "w") as w:
    for item in train_target_new:
        w.write(item)

with open('/home/hadoop-aipnlp/cephfs/data/zengweihao02/NAACL2021/Prefix-Tuning/seq2seq_adversal/data/train_adversial/train_as_test/train.oracle', "w") as w:
    for item in train_oracle_new:
        w.write(item)

with open('/home/hadoop-aipnlp/cephfs/data/zengweihao02/NAACL2021/Prefix-Tuning/seq2seq_adversal/data/train_adversial/train_as_test/train.label', "w") as w:
    for item in train_domain_label_new:
        w.write(str(item))
        w.write("\n")



with open('/home/hadoop-aipnlp/cephfs/data/zengweihao02/NAACL2021/Prefix-Tuning/seq2seq_adversal/data/train_adversial/train_as_test/val.source', "w") as w:
    for item in val_source:
        w.write(item)

with open('/home/hadoop-aipnlp/cephfs/data/zengweihao02/NAACL2021/Prefix-Tuning/seq2seq_adversal/data/train_adversial/train_as_test/val.target', "w") as w:
    for item in val_target:
        w.write(item)

with open('/home/hadoop-aipnlp/cephfs/data/zengweihao02/NAACL2021/Prefix-Tuning/seq2seq_adversal/data/train_adversial/train_as_test/val.oracle', "w") as w:
    for item in val_oracle:
        w.write(item)

with open('/home/hadoop-aipnlp/cephfs/data/zengweihao02/NAACL2021/Prefix-Tuning/seq2seq_adversal/data/train_adversial/train_as_test/val.label', "w") as w:
    for item in val_domain_label:
        w.write(str(item))
        w.write("\n")

with open('/home/hadoop-aipnlp/cephfs/data/zengweihao02/NAACL2021/Prefix-Tuning/seq2seq_adversal/data/train_adversial/train_as_test/test.source', "w") as w:
    for item in test_source:
        w.write(item)

with open('/home/hadoop-aipnlp/cephfs/data/zengweihao02/NAACL2021/Prefix-Tuning/seq2seq_adversal/data/train_adversial/train_as_test/test.target', "w") as w:
    for item in test_target:
        w.write(item)

with open('/home/hadoop-aipnlp/cephfs/data/zengweihao02/NAACL2021/Prefix-Tuning/seq2seq_adversal/data/train_adversial/train_as_test/test.oracle', "w") as w:
    for item in test_oracle:
        w.write(item)

with open('/home/hadoop-aipnlp/cephfs/data/zengweihao02/NAACL2021/Prefix-Tuning/seq2seq_adversal/data/train_adversial/train_as_test/test.label', "w") as w:
    for item in test_domain_label:
        w.write(str(item))
        w.write("\n")









print("bupt")








