# -*- coding: utf-8 -*-
from chainer import Chain, Variable, optimizers
from chainer.datasets import tuple_dataset
from chainer.training import extensions
from model import NIN,Alex,AlexLike
from data import ChainerImage
from chainer import training
import chainer.functions as F
import chainer.serializers
import chainer.links as L
import numpy as np
import datetime
import argparse
import chainer
import pickle
import glob
import re

def get_args():
    parse = argparse.ArgumentParser(description='Chainer train')
    parse.add_argument('--batchsize','-b',type=int, default=100,
                       help='Number if images in each mini batch')
    parse.add_argument('--epoch','-e',type=int, default=50,
                       help='Number of sweeps over the dataset to train')
    parse.add_argument('--gpu','-g',type=int, default=-1,
                       help='GPU ID(negative value indicates CPU')
    parse.add_argument('--out','-o', default='./result/',
                       help='Directory to output the result')
    parse.add_argument("--model","-m", default="",
                       help="Path to model")
    parse.add_argument("--name","-n", default="",
                       help="Name of model, 'NIN, Alex, AlexLike'")
    parse.add_argument('--optimizer','-O', default='')
    parse.add_argument('--size','-s', type=int, default=227,
                       help='image size')
    parse.add_argument("--classes","-c", type=int, default=0,
                       help="Input image classes")
    parse.add_argument('--gray', action="store_true", default=False)
    parse.add_argument('--img','-i', default="",
                       help='Image path for training')
    return parse.parse_args()

def log_report(log_trainer):
    trainer = log_trainer
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
    trainer.extend(extensions.dump_graph('main/loss'))
    return trainer

def main():
    args = setup_parse()
    print(args.process)

    print('# GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))

    dataset = ChainerImage(args.img)
    n_class = dataset.n_class
    train, test = dataset.train_dataset()
    print('# Training Images:',len(train))
    print('# Test Images:',len(test))
    print()

    
    model = L.Classifier(NIN(n_class))
    if args.name = "Alex":
        model = L.Classifier(Alex(n_class))
    elif args.name = "AlexLike":
        model = L.Classifier(AlexLike(n_class))

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    if args.model != '' and args.optimizer != '':
        chainer.serializers.load_npz(args.model, model)
        chainer.serializers.load_npz(args.optimizer, optimizer)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer = log_report(trainer)
    trainer.extend(extensions.ProgressBar())
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    trainer.run()
    
    date = str(datetime.datetime.today().strftime("%Y_%m_%d_%H_%M"))
    info = date + "_class_" + str(n_class)+ "_epoch_" + str(args.epoch)
    if args.gray
        info = args.name + "_gray_" + info
    else:
        info = args.name + "_color_" + info
    
    chainer.serializers.save_npz("./model/"+info+".model", model)
    chainer.serializers.save_npz("./model/"+info+".state", optimizer)

if __name__ == '__main__':
    main()
