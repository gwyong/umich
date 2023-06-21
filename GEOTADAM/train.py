import argparse, time
import numpy as np

import transform
import optimize_loss as opt_loss
from data_loader import Data_Loader

def transform_data(data, transformer):
    transform_indexes = np.tile(np.arange(transformer.n_transforms), len(data))
    transformed_data = transformer.transform_batch(np.repeat(np.array(data), transformer.n_transforms, axis=0), transform_indexes)
    return transformed_data, transform_indexes

def load_transformed_data(args, transformer):
    DL = Data_Loader()
    x_train, x_test, y_test = DL.load_dataset(args.dataset, true_label=args.class_index)
    x_train_transformed, _ = transform_data(x_train, transformer)
    x_test_transformed, _ = transform_data(x_test, transformer)
    x_test_transformed, x_train_transformed = x_test_transformed.transpose(0, 3, 1, 2), x_train_transformed.transpose(0, 3, 1, 2)
    y_test = np.array(y_test) == args.class_index
    return x_train_transformed, x_test_transformed, y_test

def load_transformed_tabular_data(args):
    DL = Data_Loader()
    x_train, normal_test, anomal_test = DL.load_dataset(args.dataset)
    y_test_f1score = np.concatenate([np.zeros(len(normal_test)), np.ones(len(anomal_test))])
    ratio = 100.0 * len(normal_test) / (len(normal_test) + len(anomal_test))

    _, n_dims = x_train.shape
    rotations = np.random.randn(256, n_dims, 32)

    x_train = np.stack([x_train.dot(rotation) for rotation in rotations], 2)
    normal_xs = np.stack([normal_test.dot(rotation) for rotation in rotations], 2)
    anomal_xs = np.stack([anomal_test.dot(rotation) for rotation in rotations], 2)
    x_test = np.concatenate([normal_xs, anomal_xs])
    return x_train, x_test, y_test_f1score, ratio

def train_anomaly_detector(args):
    if args.dataset == "cifar10" or "stl10":
        transformer = transform.get_transformer()
        x_train, x_test, y_test = load_transformed_data(args, transformer)
        J_tc = opt_loss.TransformClassifier(transformer.n_transforms, args)
        J_tc.fit_transform_classifier(x_train, x_test, y_test)
    
    if args.dataset == "thyroid":
        x_train, x_test, y_test, ratio = load_transformed_tabular_data(args)
        J_tc = opt_loss.TransformTabularClassifier(args)
        f1_score = J_tc.fit_transform_classifier(x_train, x_test, y_test, ratio)
        return f1_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Wide Residual Networks')
    parser.add_argument('--depth', default=10, type=int)
    parser.add_argument('--widen-factor', default=4, type=int)

    parser.add_argument('--batch_size', default=288, type=int)
    # parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float)
    parser.add_argument('--epochs', default=16, type=int)
    # parser.add_argument('--epochs', default=25, type=int)

    parser.add_argument('--lmbda', default=0.1, type=float)
    parser.add_argument('--reg', default=True, type=bool)
    parser.add_argument('--eps', default=0, type=float)

    parser.add_argument('--class_index', default=1, type=int)
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--analytic_margin', default=0, type=int)
    args = parser.parse_args()
    print("Dataset: ", args.dataset)
            
    if args.dataset == "cifar10" or "stl10":
        start_time = time.time()
        for i in range(10):
            args.class_index = i
            print("True Class:", args.class_index)
            train_anomaly_detector(args)
        end_time = time.time()
        print("Measured Time: ", end_time-start_time)

    if args.dataset == "thyroid":
        # python train.py --epochs=1 --batch_size=64 --dataset=thyroid --analytic_margin=1
        start_time = time.time()
        n_iters = 100
        f1_scores = np.zeros(n_iters)
        for i in range(n_iters):
            f1_scores[i] = train_anomaly_detector(args)
        print("AVG f1_score", f1_scores.mean())
        end_time = time.time()
        print("Measured Time: ", end_time-start_time)