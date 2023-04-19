BETA=0.3
DATA=FB15K
KERNEL=transe
NEG_NUM=1
MARGIN=6

CUDA_VISIBLE_DEVICES=1 nohup python run.py -dataset=$DATA \
  -num_batch=400 \
  -margin=$MARGIN \
  -neg_mode=img \
  -train_mode=adp \
  -epoch=1000 \
  -save=./checkpoint/FB15K-transe-twostage \
  -test_mode=lp \
  -img_grad=False \
  -img_dim=4096 \
  -neg_num=$NEG_NUM \
  -kernel=$KERNEL \
  -learning_rate=1.0 \
  -beta=$BETA > ./log/$DATA-$KERNEL-$MARGIN-ts-$BETA-lp.txt &
