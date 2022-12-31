cd ../../
git clone https://github.com/NLPWM-WHU/BiGCN.git
cp -f -r ./original_models/BiGCN-CKGA/* ./BiGCN
rm ./BiGCN/run_ckga.sh

cd ./BiGCN/datasets
unzip semeval14.zip -d semeval14 
cd ../

# 1. Joint Training Method
python train.py --dataset rest14 --num_layer 1 --vocab_dir datasets/semeval14/rest14_ --save True \
    --adapter_kge transh \
    --adapter_score 0 \
    --adapter_dropout 0.5 \
    --adapter_freeze_emb True \
    --adapter_norm True \
    --train_model j \
    --fuse_mode p \
    --origin_model_lr 1e-3

:<<!
# 2. If you use the original code and save the best original model, you can use Independent Training or Fine-tuning Methods
# 2.1 Independent Training Example (set orignal model learning rate to 0):
python train.py --dataset rest14 --num_layer 1 --vocab_dir datasets/semeval14/rest14_ --save True \
    --adapter_kge transh \
    --adapter_score 0 \
    --adapter_dropout 0.5 \
    --adapter_freeze_emb True \
    --adapter_norm True \
    --train_model d \ 
    --fuse_mode p \
    --origin_model_path the_best_orignal_model_path \
    --origin_model_lr 0

# 2.2 Fine-tuning Example (set a small orignal model learning rate):
python train.py --dataset rest14 --num_layer 1 --vocab_dir datasets/semeval14/rest14_ --save True \
    --adapter_kge transh \
    --adapter_score 0 \
    --adapter_dropout 0.5 \
    --adapter_freeze_emb True \
    --adapter_norm True \
    --train_model d \ 
    --fuse_mode p \
    --origin_model_path the_best_orignal_model_path \
    --origin_model_lr 1e-6 
!