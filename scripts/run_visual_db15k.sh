CUDA_VISIBLE_DEVICES=0 nohup python run.py -dataset=FB15K \
  -num_batch=400 \
  -margin=12 \
  -neg_mode=img \
  -train_mode=normal \
  -epoch=1000 \
  -save=./checkpoint/DB15K-img \
  -test_mode=lp \
  -img_grad=True \
  -kernel=transe \
  -neg_num=1 \
  -img_dim=4096 > ./log/DB15K-TransE-img.txt &
