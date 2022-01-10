with open('/home/hadoop-aipnlp/cephfs/data/zengweihao02/NAACL2021/Prefix-Tuning/seq2seq/data/multiwoz_transfer_dataset/train_as_test_new_oracle/test.source', 'r') as r:
    a = r.readlines()


dst_list = []

source_list = []

for item in a:
    item = item.split('<|endoftext|>')

    dst_list.append(item[0])
    source_list.append(item[1])




with open('/home/hadoop-aipnlp/cephfs/data/zengweihao02/NAACL2021/Prefix-Tuning/seq2seq/data/NAACL2021_Data/train_as_test/test.oracle', 'w') as w1:
    for item in dst_list:
        w1.write(item)
        w1.write('\n')

with open('/home/hadoop-aipnlp/cephfs/data/zengweihao02/NAACL2021/Prefix-Tuning/seq2seq/data/NAACL2021_Data/train_as_test/test.source', 'w') as w1:
    for item in source_list:
        w1.write(item)

print("bupt")