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
import argparse
import chainer
import pickle
import glob
import sys
import re


def get_args():
    parse = argparse.ArgumentParser(description="Chainer predict")
    parse.add_argument("--gpu","-g",type=int, default=-1,
                       help="GPU ID(Negative value indicates CPU")
    parse.add_argument("--out","-o", default="./result/",
                       help="Directory to output the result")
    parse.add_argument("--model","-m", default="",
                       help="Path to model")
    parse.add_argument("--name","-n", default="",
                       help="Name of model, 'NIN, Alex, AlexLike'")
    parse.add_argument("--size","-s", type=int, default=227,
                       help="Input image size")
    parse.add_argument("--classes","-c", type=int, default=0,
                       help="Input image classes")
    parse.add_argument("--img","-i", default="",
                       help="Image path for prediction")
    parse.add_argument("--gray", action="store_true", default=False)
    return parse.parse_args()

def is_args_ok(args):
    if args.model != "":
        print("Please set path of model, use '-m'")
    elif args.path == "":
        print("Please set path of image, use '-p'")
    elif args.classes == "":
        print("Please set number of class, use '-c'")
    elif args.name not in ["NIN","Alex","AlexLike"]:
        print("Please set name of model, use '-n'")
    else:
        return True
    return False

def predict(model, dataset, args.gpu):
    img, label = dataset
    if args.gpu >= 0:
        img = model.xp.asarray(img)
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        y = model(x)
    if args.gpu >= 0:
        y = to_cpu(y.array)
    print("Predict : ", y.argmax(axis=1)[0], "Answer : ", label)
    if y.argmax(axis=1)[0] == label:
        return 1
    else:
        return 0

def main():
    args = get_args()
    if not is_args_ok(args):
        sys.exit(-1)
    model = NIN(args.classes)
    if args.name == "Alex":
        model = Alex(args.classes)
    elif args.name == "AlexLike":
        model = AlexLike(args.classes)
    serializers.load_npz(args.model, model)
    if args.gpu >= 0:
        model.to_gpu(args.gpu)
    imgdata = ChainerImage(args.path, args.classes)
    p_imgs = imgdata.prediction_dataset()
    correct, total = 0, 0
    for dataset in p_imgs:
        correct += predict(model, dataset, args.gpu)
        total += 1
    print("Accuracy : ", correct*1.0/total)
    return

if __name__ == "__main__":
    main()
