MARGIN=12
KERNEL=transe

CUDA_VISIBLE_DEVICES=0 nohup python run.py -dataset=DB15K \
  -num_batch=400 \
  -margin=$MARGIN \
  -neg_mode=normal \
  -train_mode=normal \
  -epoch=1000 \
  -save=./checkpoint/FB15K-base-$KERNEL-margin$MARGIN \
  -test_mode=lp \
  -img_grad=False \
  -img_dim=4096 \
  -neg_num=1 \
  -kernel=transe \
  -learning_rate=1.0 > ./log/DB15K-base-$KERNEL-margin$MARGIN.txt &