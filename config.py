import argparse



def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_prop', type=float, default=0.5)
    parser.add_argument('--valid_prop', type=float, default=0.25)
    parser.add_argument('--dname', default='benchmark1')
    parser.add_argument('--inside_model', default='gin')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--display_step', default=1, type=int)
    parser.add_argument('--runs', default=5, type=int)
    parser.add_argument('--train_patience', default=10, type=int)
    parser.add_argument('--cuda', default=3, choices=[0, 1, 2, 3], type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--GIN_num_layers', default=2, type=int)
    parser.add_argument('--GIN_hidden_dim', default=8, type=int)
    parser.add_argument('--output_dim', default=8, type=int)
    parser.add_argument('--aggregate', default='add', choices=['add', 'mean'])
    parser.add_argument('--numK', default=10, type=int)
    parser.add_argument('--numL', default=10, type=int)
    parser.add_argument('--reg_lambda', default=0.05, type=float)
    parser.add_argument('--weight_decay', default=0.001, type=float)
    parser.add_argument('--prop', default=0.1, type=float)
    return parser.parse_args()
