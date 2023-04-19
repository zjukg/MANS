import argparse


def get_args():
    arg = argparse.ArgumentParser()
    arg.add_argument('-dataset', type=str, default='FB15K')
    arg.add_argument('-num_batch', type=int, default=400)
    arg.add_argument('-margin', type=float, default=5.0)
    arg.add_argument('-neg_mode', type=str, default='img', choices=['img', 'ent', 'normal', 'hybrid', 'adaptive'])
    arg.add_argument('-train_mode', type=str, default='normal', choices=['normal', 'adp'])
    arg.add_argument('-epoch', type=int, default=1000)
    arg.add_argument('-save', type=str)
    arg.add_argument('-test_mode', type=str, default='lp')
    arg.add_argument('-img_dim', type=int, default=4096)
    arg.add_argument('-img_grad', type=bool, default=False)
    arg.add_argument('-kernel', type=str, default='transe', required=True, choices=['transe', 'dismult', 'rotate'])
    arg.add_argument('-neg_num', type=int, default=1)
    arg.add_argument('-loss_type', type=str, default='normal', choices=['normal', 'adv'])
    arg.add_argument('-learning_rate', type=float, default=0.1)
    arg.add_argument('-beta', type=float, default=0.5)
    arg.add_argument('-adv_temp', type=float, default=2.0)
    return arg.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(args)
