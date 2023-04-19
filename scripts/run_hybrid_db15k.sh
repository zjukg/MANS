BETA=0.3
TASK=lp
DATA=DB15K
EPOCH=1000
GPU=2

CUDA_VISIBLE_DEVICES=$GPU nohup python run.py -dataset=$DATA \
  -num_batch=400 \
  -margin=12 \
  -neg_mode=hybrid \
  -train_mode=normal \
  -epoch=1000 \
  -save=./checkpoint/$DATA-transe-hybrid-$BETA-$EPOCH \
  -test_mode=$TASK \
  -img_grad=False \
  -img_dim=4096 \
  -neg_num=1 \
  -kernel=transe \
  -learning_rate=1.0 \
  -beta=$BETA > ./log/$DATA-transe-hybrid-$BETA-$EPOCH.txt &
