dname=$1
method=GIN
lr=0.01
wd=0.05
numL=5
numK=5
GIN_num_layers=2
GIN_hidden_dim=8
cuda=0
reg_lambda=0.001
runs=5
epochs=500
prop=0.005



if [ "$method" = "GIN" ]; then
    echo =============
    python for_testing_prop.py \
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
        --prop $prop \
        --lr $lr
fi