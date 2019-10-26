You can download epsilon dataset from: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
Then generate their TFRecords by:
```cli
python epsilon2tfrecords.py --data-dir=<Where your downloaded data resides>
```

Then you can run LUPA-SGD using this script:

```cli
python main.py --data-dir=./data/epsilon \
               --variable-strategy GPU \
               --learning-rate=0.01 \
               --job-dir=./tmp/gpu4/epsilon_multi_91_batch_512-adaptive2 \
               --train-steps=5469 \
               --num-gpus=4 \
               --run-type multi \
               --redundancy=-0.25 \
               --sync-step=91 \
               --problem epsilon \
               --train-batch-size 128
```

where sync-step is equal to $tau$ in paper. If you want to have adaptive number of synchronization use `--adaptive` and ensure to have number of communication set to the desired number by `--num-comm`, as well.
