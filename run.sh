dname=$1
method=$2
lr=$3
wd=0.05
numL=$4
numK=$5
GIN_num_layers=2
GIN_hidden_dim=8
cuda=0
reg_lambda=$6
runs=5
epochs=500

if [ "$method" = "GIN" ]; then
    echo =============
    python train.py \
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