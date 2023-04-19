MARGIN=6
KERNEL=transe

CUDA_VISIBLE_DEVICES=1 nohup python run.py -dataset=FB15K \
  -num_batch=400 \
  -margin=$MARGIN \
  -neg_mode=normal \
  -train_mode=normal \
  -epoch=1200 \
  -save=./checkpoint/FB15K-base-$KERNEL-margin$MARGIN \
  -test_mode=lp \
  -img_grad=False \
  -img_dim=4096 \
  -neg_num=1 \
  -kernel=transe \
  -learning_rate=1.0 > ./log/FB15K-base-$KERNEL-margin$MARGIN.txt &