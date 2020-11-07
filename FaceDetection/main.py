import training
import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', '-m', choices = ['train', 'test'], help = "This is for choosing the mode of this model. Default is Test")
    parser.add_argument('--image_size', type = int, default = 416, help = "The common width and height for all images")
    parser.add_argument('--batch_size', type = int, default = 10)
    parser.add_argument('--epochs', type = int, default = 100)
    parser.add_argument('--coord_scale', type = float, default = 5)
    parser.add_argument('--noobj_scale', type = float, default = 0.5)
    parser.add_argument('--dataset_path', type = str, default = 'data')
    parser.add_argument('--model_path', type = str, default = 'data\\models')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    if( args.mode == 'train'):
        training.train(args)

    else:
        print("TODO")

