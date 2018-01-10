#!/usr/bin/python
#-*- coding: utf-8 -*-

from mxnet import gluon
from mxnet.gluon import nn
from mxnet import ndarray as nd
from mxnet import init
from mxnet import image
from mxnet import autograd as ag
import numpy as np
import time
import mxnet as mx
import json
import sys, datetime
from matplotlib import pyplot as plt

num_classes = 2

def try_gpu():
    ctx = mx.gpu()
    try:
        _ = nd.zeros((1, ), ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx
ctx = try_gpu()

class DataLoader(object):
    def __init__(self, dataset, batch_size, shuffle=True, resize=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.resize = resize

    def __iter__(self):
        dataset = self.dataset[:]
        X = dataset[0]
        y = nd.array(dataset[1])
        n = X.shape[0]
        resize = self.resize
        if self.shuffle:
            idx = np.arange(n)
            np.random.shuffle(idx)
            X = nd.array(X.asnumpy()[idx])
            y = nd.array(y.asnumpy()[idx])
        for i in range(n//self.batch_size):
            batch_x = X[i*self.batch_size: (i+1)*self.batch_size]
            if resize:
                new_data = nd.zeros(shape=(batch_x.shape[0], resize, resize, batch_x.shape[3]))
                for j in range(batch_x.shape[0]):
                    new_data[j] = image.imresize(batch_x[j], resize, resize)
                batch_x = new_data
            batch_x = nd.transpose(batch_x, axes=(0, 3, 1, 2))
            yield (batch_x, y[i*self.batch_size: (i+1)*self.batch_size])

    def __len__(self):
        return len(self.dataset[0])//self.batch_size

class Residual(gluon.nn.Block):
    def __init__(self, channels, same_shape=True, is_dropout=False, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.same_shape = same_shape
        self.is_dropout = is_dropout
        strides = 1 if same_shape else 2
        self.conv_1 = gluon.nn.Conv2D(channels=channels, kernel_size=3, padding=1, strides=strides)
        self.conv_2 = gluon.nn.Conv2D(channels=channels, kernel_size=3, padding=1)
        self.bn_1 = gluon.nn.BatchNorm(axis=1)
        self.bn_2 = gluon.nn.BatchNorm(axis=1)
        if not same_shape:
            self.conv_3 = gluon.nn.Conv2D(channels=channels, kernel_size=1, strides=strides)
        if is_dropout:
            self.dropout = gluon.nn.Dropout(0.5)

    def forward(self, x):
        out = self.conv_1(nd.relu(self.bn_1(x)))
        out = nd.relu(self.bn_2(out))
        if self.is_dropout:
            out = self.dropout(out)
        out = self.conv_2(out)
        if not self.same_shape:
            x = self.conv_3(x)
        return out + x

class Resnet(gluon.nn.Block):
    def __init__(self, **kwargs):
        super(Resnet, self).__init__(**kwargs)
        with self.name_scope():
            b1 = gluon.nn.Conv2D(channels=32, kernel_size=5, strides=2)

            b2 = gluon.nn.Sequential()
            b2.add(
                gluon.nn.MaxPool2D(pool_size=2),
                Residual(channels=32),
                Residual(channels=32)
            )

            b3 = gluon.nn.Sequential()
            b3.add(
                Residual(channels=64, same_shape=False),
                Residual(channels=64)
            )

            # b4 = gluon.nn.Sequential()
            # b4.add(
            #     Residual(channels=64, same_shape=False),
            #     Residual(channels=64)
            # )

            b4 = gluon.nn.Sequential()
            b4.add(
                Residual(channels=128, same_shape=False),
                Residual(channels=128)
            )

            # b6 = gluon.nn.Sequential()
            # b6.add(
            #     Residual(channels=128, same_shape=False),
            #     Residual(channels=128, is_dropout=True)
            # )

            # b6 = gluon.nn.Sequential()
            # b6.add(
            #     gluon.nn.Dense(256, activation='relu'),
            #     gluon.nn.BatchNorm(axis=1),
            #     gluon.nn.Dropout(0.5),
            #     gluon.nn.Dense(256, activation='relu'),
            #     gluon.nn.BatchNorm(axis=1),
            #     gluon.nn.Dropout(0.5),
            #     gluon.nn.Dense(num_classes)
            # )

            b5 = gluon.nn.Sequential()
            b5.add(
                Residual(channels=256, same_shape=False),
                Residual(channels=256)
            )

            b6 = gluon.nn.Sequential()
            b6.add(
                gluon.nn.AvgPool2D(pool_size=2),
                gluon.nn.Dense(num_classes, activation='sigmoid')
            )
            self.net = gluon.nn.Sequential()
            self.net.add(b1, b2, b3, b4, b5, b6)

    def forward(self, x):
        return self.net(x)

# net = Resnet()
net = nn.Sequential()
with net.name_scope():
    net.add(
        nn.Conv2D(channels=32, kernel_size=3, padding=1),
        nn.BatchNorm(axis=1),
        nn.Activation(activation='relu'),
        nn.Dropout(0.2),
        nn.Conv2D(channels=32, kernel_size=3, padding=1),
        nn.BatchNorm(axis=1),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        nn.Dropout(0.5),

        nn.Conv2D(channels=64, kernel_size=3, padding=1),
        nn.BatchNorm(axis=1),
        nn.Activation(activation='relu'),
        nn.Dropout(0.2),
        nn.Conv2D(channels=64, kernel_size=3, padding=1),
        nn.BatchNorm(axis=1),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        nn.Dropout(0.5),

        nn.Conv2D(channels=128, kernel_size=3, padding=1),
        nn.BatchNorm(axis=1),
        nn.Activation(activation='relu'),
        nn.Dropout(0.2),
        nn.Conv2D(channels=128, kernel_size=3, padding=1),
        nn.BatchNorm(axis=1),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2),
        nn.Dropout(0.5),

        nn.Conv2D(channels=128, kernel_size=3, padding=1),
        nn.BatchNorm(axis=1),
        nn.Activation(activation='relu'),
        nn.Dropout(0.2),
        nn.Conv2D(channels=128, kernel_size=3, padding=1),
        nn.BatchNorm(axis=1),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        nn.Dropout(0.5),

        nn.Flatten(),

        nn.Dense(256),
        nn.BatchNorm(axis=1),
        nn.Activation(activation='relu'),
        nn.Dropout(0.5),

        nn.Dense(256),
        nn.BatchNorm(axis=1),
        nn.Activation(activation='relu'),
        nn.Dropout(0.5),
        nn.Dense(2)
    )
net.initialize(init=init.Xavier(), ctx=ctx)
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.001})

def softmax(X):
    X_max = nd.max(X, axis=1, keepdims=True)
    X = X - X_max
    exp = nd.exp(X)
    partition = nd.sum(exp, axis=1, keepdims=True)
    return exp / partition

def cross_entropy(yhat, y):
    return - nd.log(nd.pick(yhat, y, axis=1, keepdims=True))

def accuracy(output, label):
    return nd.mean(output.argmax(axis=1)==label).asscalar()

def evaluate_accuracy(data_iter):
    acc = .0
    total_loss = .0
    for data, label in data_iter:
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        prob = softmax(output)
        loss = cross_entropy(prob, label)
        total_loss += nd.mean(loss).asscalar()
        acc += accuracy(output, label)
    return acc / len(data_iter), total_loss / len(data_iter)

def gen_2channel_img():
    with open('./input/train.json') as f:
        data = json.load(f)

        for img in data:

            name = img['id']
            label = img['is_iceberg']
            band_1 = nd.array(img['band_1']).reshape(( 75, 75))
            imageio.imwrite('ice_img/%s_%s_hh.jpg' % (label, name), band_1.asnumpy())

            band_2 = nd.array(img['band_2']).reshape((75, 75))
            imageio.imwrite('ice_img/%s_%s_hv.jpg' % (label, name), band_2.asnumpy())

def apply_aug_list(img, augs):
    for f in augs:
        img = f(img)
    return img

def resize(x, resize):
    new_data = nd.zeros(shape=(x.shape[0], resize, resize, x.shape[3]))
    for j in range(x.shape[0]):
        new_data[j] = image.imresize(x[j], resize, resize)
    return new_data

def transform(data, aug):
    data = data.astype('float32')
    if aug is not None:
        data = nd.stack(*[aug(d) for d in data])
    return data

def train(train_data, test_data, batch_size):
    epoches = 100
    for e in range(epoches):
        total_loss = .0
        train_acc = .0
        start = time.time()
        batch = 0
        for data, label in train_data:
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            with ag.record():
                net_out = net(data)
                output = softmax(net_out)
                loss = cross_entropy(output, label)
            loss.backward()
            trainer.step(batch_size)
            train_acc += accuracy(output, label)
            total_loss += nd.mean(loss).asscalar()
            if batch % 10 == 0:
                print('batch %d tarin done.' % batch)
                sys.stdout.flush()
            batch += 1
        test_acc, test_loss = evaluate_accuracy(test_data)
        print("e: %d, train_loss: %f, train_acc: %f, test_acc: %f, test_loss: %f, cost_time: %d" % (e, total_loss / len(train_data), \
              train_acc / len(train_data), test_acc, test_loss, time.time()- start))

def augment_data(imags, label):
    datas, labels = [], []
    n = imags.shape[0]
    for k in range(n):
        imag = imags[k].reshape(shape=(1, imags[k].shape[0], imags[k].shape[1], imags[k].shape[2]))
        for i in range(4):
            trans_band = transform(imag, image.HorizontalFlipAug(.5))
            trans_band = trans_band.astype('float32')
            max_val = nd.max(trans_band)
            min_val = nd.min(trans_band)
            datas.append((trans_band - min_val) / (max_val - min_val))
            labels.append(label[k].asscalar())

        for i in range(9):
            trans_band = transform(imag, image.RandomSizedCropAug((75, 75), .75, (.8, 1.2)))
            trans_band = trans_band.astype('float32')
            max_val = nd.max(trans_band)
            min_val = nd.min(trans_band)
            datas.append((trans_band - min_val) / (max_val - min_val))
            labels.append(label[k].asscalar())
        # brightness augmenter
        for i in range(9):
            trans_band = transform(imag, image.BrightnessJitterAug(.1))
            trans_band = trans_band.astype('float32')
            max_val = nd.max(trans_band)
            min_val = nd.min(trans_band)
            datas.append((trans_band - min_val) / (max_val - min_val))
            labels.append(label[k].asscalar())
        # random crop augmenter
        for i in range(9):
            trans_band = resize(transform(imag, image.RandomCropAug((50,50))), 75)
            trans_band = trans_band.astype('float32')
            max_val = nd.max(trans_band)
            min_val = nd.min(trans_band)
            datas.append((trans_band - min_val) / (max_val - min_val))
            labels.append(label[k].asscalar())
        # center crop augmenter
        trans_band = resize(transform(imag, image.CenterCropAug((50,50))), 75)
        trans_band = trans_band.astype('float32')
        max_val = nd.max(trans_band)
        min_val = nd.min(trans_band)
        datas.append((trans_band - min_val) / (max_val - min_val))
        labels.append(label[k].asscalar())
    ds = nd.concat(*datas, dim=0)
    # if is_train:
    #     max_val = nd.max(ds.astype('float32'))
    #     min_val = nd.min(ds.astype('float32'))
    # ds = (ds.astype('float32') - min_val) / (max_val - min_val)
    # print("max val after normalizing:", nd.max(ds.astype('float32')))
    # print("min val after normalizing:", nd.min(ds.astype('float32')))
    return ds, labels

if __name__ == '__main__':
    datas, labels = [], []
    cnt = 0
    with open('input/train.json') as f:
        data = json.load(f)
        for img in data:
            name = img['id']
            label = img['is_iceberg']
            band_1 = nd.array(img['band_1']).reshape((1, 75, 75, 1))
            band_2 = nd.array(img['band_2']).reshape((1, 75, 75, 1))
            band = nd.concat(band_1, band_2, dim=3)
            datas.append(band)
            labels.append(label)
    ds = nd.concat(*datas, dim=0)
    print("finish load data")

    num = ds.shape[0]
    idx = np.arange(num)
    # np.random.shuffle(idx)
    split = num // 4
    test_idx = idx[:split]
    train_idx = idx[split:]
    print(train_idx.shape)
    print(test_idx.shape)
    sys.stdout.flush()

    train_ds = (
        nd.array(ds.asnumpy()[train_idx]).astype('float32'),
        nd.array(np.array(labels)[train_idx]).astype('float32')
        )
    test_ds = (
        nd.array(ds.asnumpy()[test_idx]).astype('float32'),
        nd.array(np.array(labels)[test_idx]).astype('float32')
        )
    train_ds_aug, label_train_aug = augment_data(train_ds[0], train_ds[1])
    # print(label_train_aug)
    train_ds = (
        train_ds_aug,
        nd.array(label_train_aug).astype('float32')
        )
    test_ds_aug, label_test_aug = augment_data(test_ds[0], test_ds[1])
    test_ds = (
        test_ds_aug,
        nd.array(label_test_aug).astype('float32')
        )
    batch_size = 128
    train_data = DataLoader(train_ds, batch_size, shuffle=True)
    test_data = DataLoader(test_ds, batch_size, shuffle=False)
    print(len(test_ds[0]))
    sys.stdout.flush()
    train(train_data, test_data, batch_size)
