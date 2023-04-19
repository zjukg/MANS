BETA=0
TASK=lp
DATA=DB15K
NUM_BATCH=400
KERNEL=transe
MARGIN=12
MODE=adaptive
EPOCH=1000
GPU=1

CUDA_VISIBLE_DEVICES=$GPU nohup python run.py -dataset=$DATA \
  -num_batch=$NUM_BATCH \
  -margin=$MARGIN \
  -neg_mode=$MODE \
  -train_mode=normal \
  -epoch=$EPOCH \
  -save=./checkpoint/$DATA-$KERNEL-$MODE-$BETA-$NUM_BATCH-$MARGIN-$EPOCH \
  -test_mode=$TASK \
  -img_grad=False \
  -img_dim=4096 \
  -neg_num=1 \
  -kernel=$KERNEL \
  -learning_rate=1.0 \
  -beta=$BETA > ./log/$DATA-$KERNEL-$MODE-$BETA-$NUM_BATCH-$MARGIN-$EPOCH.txt &
