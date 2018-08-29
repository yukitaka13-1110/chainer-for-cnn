from chainer import initializers as I
from chainer import Chain, Variable
import chainer.functions as F
import chainer.links as L
import numpy as np
import chainer

class Alex(chainer.Chain):
    insize = 227
    def __init__(self, n_out):
        super(Alex,self).__init__()
        with self.init_scope():
            self.conv1=L.Convolution2D(None,  96, 11, stride=4)
            self.conv2=L.Convolution2D(None, 256,  5, pad=2)
            self.conv3=L.Convolution2D(None, 384,  3, pad=1)
            self.conv4=L.Convolution2D(None, 384,  3, pad=1)
            self.conv5=L.Convolution2D(None, 256,  3, pad=1)
            self.fc6=L.Linear(None, 4096)
            self.fc7=L.Linear(None, 4096)
            self.fc8=L.Linear(None, n_out)

    def __call__(self, x):
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.dropout(F.relu(self.fc6(h)))
        h = F.dropout(F.relu(self.fc7(h)))
        h = self.fc8(h)
        return h


class NIN(chainer.Chain):

    """Network-in-Network example model."""
    def __init__(self, n_out):
        super(NIN, self).__init__()
        conv_init = I.HeNormal()  # MSRA scaling
        self.n_out = n_out
        with self.init_scope():
            self.mlpconv1 = L.MLPConvolution2D(
                None, (96, 96, 96), 11, stride=4, conv_init=conv_init)
            self.mlpconv2 = L.MLPConvolution2D(
                None, (256, 256, 256), 5, pad=2, conv_init=conv_init)
            self.mlpconv3 = L.MLPConvolution2D(
                None, (384, 384, 384), 3, pad=1, conv_init=conv_init)
            self.mlpconv4 = L.MLPConvolution2D(
                None, (1024, 1024, self.n_out), 3, pad=1, conv_init=conv_init)

    def __call__(self, x):
        h = F.max_pooling_2d(F.relu(self.mlpconv1(x)), 3, stride=2)
        h = F.max_pooling_2d(F.relu(self.mlpconv2(h)), 3, stride=2)
        h = F.max_pooling_2d(F.relu(self.mlpconv3(h)), 3, stride=2)
        h = self.mlpconv4(F.dropout(h))
        h = F.reshape(F.average_pooling_2d(h, 6), (len(x), self.n_out))
        return h


class AlexLike(chainer.Chain):

    """Single-GPU AlexNet without partition toward the channel axis."""
    def __init__(self, n_out):
        super(AlexLike, self).__init__()
        with self.init_scope():
            self.conv1=L.Convolution2D(None,  96, 11, stride=4)
            self.conv2=L.Convolution2D(None, 256,  5, pad=2)
            self.conv3=L.Convolution2D(None, 384,  3, pad=1)
            self.conv4=L.Convolution2D(None, 384,  3, pad=1)
            self.conv5=L.Convolution2D(None, 256,  3, pad=1)
            self.fc6=L.Linear(None, 4096)
            self.fc7=L.Linear(None, 1024)
            self.fc8=L.Linear(None, n_out)

    def __call__(self, x):
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.dropout(F.relu(self.fc6(h)))
        h = F.dropout(F.relu(self.fc7(h)))
        h = self.fc8(h)
        return h


class FromCaffeAlexnet(chainer.Chain):
    insize = 128
    def __init__(self, n_out):
        super(FromCaffeAlexnet, self).__init__()
        with self.init_scope():
            self.conv1=L.Convolution2D(None, 96, 11, stride=2)
            self.conv2=L.Convolution2D(None, 256, 5, pad=2)
            self.conv3=L.Convolution2D(None, 384, 3, pad=1)
            self.conv4=L.Convolution2D(None, 384, 3, pad=1)
            self.conv5=L.Convolution2D(None, 256, 3, pad=1)
            self.my_fc6=L.Linear(None, 4096)
            self.my_fc7=L.Linear(None, 1024)
            self.my_fc8=L.Linear(None, n_out)

    def __call__(self, x):
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.dropout(F.relu(self.my_fc6(h)))
        h = F.dropout(F.relu(self.my_fc7(h)))
        h = self.my_fc8(h)
        return h


class GoogLeNet(chainer.Chain):
    insize = 128
    def __init__(self, n_out):
        super(GoogLeNet, self).__init__()
        with self.init_scope():
            self.conv1=L.Convolution2D(None,  64, 7, stride=1, pad=3)
            self.conv2_reduce=L.Convolution2D(None,  64, 1)
            self.conv2=L.Convolution2D(None, 192, 3, stride=1, pad=1)
            self.inc3a=L.Inception(None,  64,  96, 128, 16,  32,  32)
            self.inc3b=L.Inception(None, 128, 128, 192, 32,  96,  64)
            self.inc4a=L.Inception(None, 192,  96, 208, 16,  48,  64)
            self.inc4b=L.Inception(None, 160, 112, 224, 24,  64,  64)
            self.inc4c=L.Inception(None, 128, 128, 256, 24,  64,  64)
            self.inc4d=L.Inception(None, 112, 144, 288, 32,  64,  64)
            self.inc4e=L.Inception(None, 256, 160, 320, 32, 128, 128)
            self.inc5a=L.Inception(None, 256, 160, 320, 32, 128, 128)
            self.inc5b=L.Inception(None, 384, 192, 384, 48, 128, 128)
            self.loss3_fc=L.Linear(None, n_out)

            self.loss1_conv=L.Convolution2D(None, 128, 1)
            self.loss1_fc1=L.Linear(None, 1024)
            self.loss1_fc2=L.Linear(None, n_out)

            self.loss2_conv=L.Convolution2D(None, 128, 1)
            self.loss2_fc1=L.Linear(None, 1024)
            self.loss2_fc2=L.Linear(None, n_out)

    def __call__(self, x, t):
        h = F.relu(self.conv1(x))
        h = F.local_response_normalization(
            F.max_pooling_2d(h, 3, stride=2), n=5)
        h = F.relu(self.conv2_reduce(h))
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(
            F.local_response_normalization(h, n=5), 3, stride=2)

        h = self.inc3a(h)
        h = self.inc3b(h)
        h = F.max_pooling_2d(h, 3, stride=2)
        h = self.inc4a(h)

        l = F.average_pooling_2d(h, 5, stride=3)
        l = F.relu(self.loss1_conv(l))
        l = F.relu(self.loss1_fc1(l))
        l = self.loss1_fc2(l)
        loss1 = F.softmax_cross_entropy(l, t)

        h = self.inc4b(h)
        h = self.inc4c(h)
        h = self.inc4d(h)

        l = F.average_pooling_2d(h, 5, stride=3)
        l = F.relu(self.loss2_conv(l))
        l = F.relu(self.loss2_fc1(l))
        l = self.loss2_fc2(l)
        loss2 = F.softmax_cross_entropy(l, t)

        h = self.inc4e(h)
        h = F.max_pooling_2d(h, 3, stride=2)
        h = self.inc5a(h)
        h = self.inc5b(h)

        h = F.average_pooling_2d(h, 7, stride=1)
        h = self.loss3_fc(F.dropout(h, 0.4))
        loss3 = F.softmax_cross_entropy(h, t)

        loss = 0.3 * (loss1 + loss2) + loss3
        accuracy = F.accuracy(h, t)

        chainer.report({
            'loss': loss,
            'loss1': loss1,
            'loss2': loss2,
            'loss3': loss3,
            'accuracy': accuracy
        }, self)
        return loss

def predict(model, valData):
    x = Variable(valData)
    y = F.softmax(model.predictor(x.data[0]))
    return y.data[0]

def predictNotPredictor(model, valData):
    x = Variable(valData)
    y = model.model(x)
    return y.data[0]
