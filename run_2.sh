dname=$1
method=GIN
lr=0.01
wd=0.05
numL=5
numK=5
GIN_num_layers=2
GIN_hidden_dim=8
cuda=0
reg_lambda=0.01
runs=5
epochs=500

if [ "$method" = "GIN" ]; then
    echo =============
    CUDA_VISIBLE_DEVICES=2 python train.py \
        --dname $dname \
        --GIN_num_layers $GIN_num_layers \
        --GIN_hidden_dim $GIN_hidden_dim \
        --weight_decay $wd \
        --numL $numL \
        --numK $numK \
        --reg_lambda $reg_lambda \
        --epochs $epochs \
        --runs $runs \
        --cuda $cuda \
        --lr $lr
fi