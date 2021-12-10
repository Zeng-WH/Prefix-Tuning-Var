# DOP-Tuning

DOP-Tuning(Domain-Oriented Prefix-tuning model)代码基于[Prefix-Tuning](https://github.com/XiangLi1999/PrefixTuning)改进.

## Files

```
├── seq2seq                       # Code for encoder-decoder architecture
│   ├── train_bart.py             # high-level scripts to train.
│   ├── prefixTuning.py           # code that implements prefix-tuning.
│   ├── finetune.py               # training code (contains data loading, model loading, and calls trainer)   
│   ├── lightning_base.py         # helper code
│   ├── utils.py                  # helper code
│   └── callbacks.py              # helper code
└── ...
```

To run the code for encoder-decoder architecture like BART, the code is in `seq2seq`. This corresponds to the summarization experiments in the paper.

## Setup:

```
cd transformer; pip install -e .
```

## Train via DOP-tuning

```shell
cd seq2seq; 

python train_bart.py --mode xsum --preseqlen 200 --do_train yes --fp16 yes --bsz 16  --epoch 30  --gradient_accumulation_step 3 --learning_rate 0.00005  --mid_dim 800 --use_lowdata_token 'yes' --lowdata_token 'summarize'
```

其中`use_lowdata_token`表示是否采用domain word初始化的方式；`lowdata_token`表示传入的domain word.

### Decode

```shell
cd seq2seq; 

python train_bart.py --mode xsum --do_train no --prefix_model_path {checkpoint_path} --preseqlen {same as training} --mid_dim {same as training} --use_lowdata_token 'yes' --lowdata_token 'summarize'
```

