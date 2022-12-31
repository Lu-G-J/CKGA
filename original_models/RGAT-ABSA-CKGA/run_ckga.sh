cd ../../
git clone https://github.com/shenwzh3/RGAT-ABSA.git
cp -f ./original_models/RGAT-ABSA-CKGA/* ./RGAT-ABSA
rm ./RGAT-ABSA/run_ckga.sh


cd ./RGAT-ABSA
# 1. Joint Training Method
python run.py --dataset_name rest --output_dir data/output-gcn --gat_our --highway --num_heads 2 --dropout 0.8 --per_gpu_train_batch_size 32 --hidden_size 512 --final_hidden_size 400 \
    --train_model j \
    --adapter_gcn_out_dim 100 \
    --logging_steps 3 \
    --origin_model_lr 1e-3  \
    --learning_rate 1e-4 \
    --fuse_mode c

:<<!
# 2. If you use the original code and save the best original model, you can use Independent Training or Fine-tuning Methods
# 2.1 Independent Training Example (set orignal model learning rate to 0):
python run.py --dataset_name rest --output_dir data/output-gcn --gat_our --highway --num_heads 2 --dropout 0.8 --per_gpu_train_batch_size 32 --hidden_size 512 --final_hidden_size 400 \
    --train_model d \
    --origin_model_path the_best_orignal_model_path \
    --adapter_gcn_out_dim 50 \
    --logging_steps 3 \
    --origin_model_lr 0  \
    --learning_rate 1e-3 \
    --fuse_mode c \
    --adapter_dropout 0.5 &

# 2.2 Fine-tuning Example (set a small orignal model learning rate):
python run.py --dataset_name rest --output_dir data/output-gcn --gat_our --highway --num_heads 2 --dropout 0.8 --per_gpu_train_batch_size 32 --hidden_size 512 --final_hidden_size 400 \
    --train_model d \
    --origin_model_path the_best_orignal_model_path \
    --adapter_gcn_out_dim 50 \
    --logging_steps 3 \
    --origin_model_lr 1e-6  \
    --learning_rate 1e-3 \
    --fuse_mode c \
    --adapter_dropout 0.5 &
!
