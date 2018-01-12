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

class TestDataLoader(object):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        dataset = self.dataset[:]
        X = dataset[0]
        y = dataset[1]
        z = nd.array(dataset[2])
        n = X.shape[0]
        last_i = 0
        for i in range(n//self.batch_size):
            last_i = i
            batch_x = X[i*self.batch_size: (i+1)*self.batch_size]
            batch_x = nd.transpose(batch_x, axes=(0, 3, 1, 2))
            yield (batch_x, y[i*self.batch_size: (i+1)*self.batch_size], z[i*self.batch_size: (i+1)*self.batch_size])
        batch_x = X[(last_i+1)*self.batch_size:]
        batch_x = nd.transpose(batch_x, axes=(0, 3, 1, 2))
        yield (batch_x, y[(last_i+1)*self.batch_size: ], z[(last_i+1)*self.batch_size: ])

    def __len__(self):
        return len(self.dataset[0])//self.batch_size

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
        z = nd.array(dataset[2])
        n = X.shape[0]
        resize = self.resize
        if self.shuffle:
            idx = np.arange(n)
            np.random.shuffle(idx)
            X = nd.array(X.asnumpy()[idx])
            y = nd.array(y.asnumpy()[idx])
            z = nd.array(z.asnumpy()[idx])
        for i in range(n//self.batch_size):
            batch_x = X[i*self.batch_size: (i+1)*self.batch_size]
            if resize:
                new_data = nd.zeros(shape=(batch_x.shape[0], resize, resize, batch_x.shape[3]))
                for j in range(batch_x.shape[0]):
                    new_data[j] = image.imresize(batch_x[j], resize, resize)
                batch_x = new_data
            batch_x = nd.transpose(batch_x, axes=(0, 3, 1, 2))
            yield (batch_x, y[i*self.batch_size: (i+1)*self.batch_size], z[i*self.batch_size: (i+1)*self.batch_size])

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
            self.conv_3 = gluon.nn.Conv2D(channels=channels, kernel_size=3, padding=1, strides=strides)
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
                Residual(channels=512, same_shape=False),
                Residual(channels=512)
            )

            b7 = gluon.nn.Sequential()
            b7.add(
                gluon.nn.AvgPool2D(pool_size=3),
                gluon.nn.Dense(num_classes, activation='sigmoid')
            )
            self.net = gluon.nn.Sequential()
            self.net.add(b1, b2, b3, b4, b5, b6, b7)

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
        nn.Dropout(0.5)
    )
net.initialize(init=init.Xavier(), ctx=ctx)
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.001})

weight_scale = 0.01
weight = nd.random_normal(shape=(256, 2), scale=weight_scale, ctx=ctx)
angle_weight = nd.ones(shape=(1, 2), ctx=ctx)
bias = nd.zeros(shape=(2, ), ctx=ctx)
params = [weight, angle_weight, bias]
for param in params:
    param.attach_grad()
weight_v = nd.zeros(shape=(256, 2), ctx=ctx)
angle_weight_v = nd.zeros(shape=(1, 2), ctx=ctx)
bias_v = nd.zeros(shape=(2, ), ctx=ctx)
vs = [weight_v, angle_weight_v, bias_v]
weight_sqr = nd.zeros(shape=(256, 2), ctx=ctx)
angle_weight_sqr = nd.zeros(shape=(1, 2), ctx=ctx)
bias_sqr = nd.zeros(shape=(2, ), ctx=ctx)
sqrs = [weight_sqr, angle_weight_sqr, bias_sqr]

beta1 = 0.9
beta2 = 0.999
eps_stable = 1e-8
def adam(params, vs, sqrs, lr, batch_size, t):
    for param, v, sqr in zip(params, vs, sqrs):
        current_v = beta1 * v + (1 - beta1) * param.grad
        current_sqr = beta2 * sqr + (1 - beta2) * param.grad * param.grad
        v[:] = current_v / (1 - beta1**t)
        sqr[:] = current_sqr / (1 - beta2**t)
        grad = v / nd.sqrt(sqr + eps_stable)
        param[:] = param - (lr / batch_size) * grad

def add_angle(X, angle):
    out = nd.dot(X, weight) + nd.dot(angle.reshape(shape=(-1, 1)), angle_weight) + bias
    return out

def softmax(X):
    X_max = nd.max(X, axis=1, keepdims=True)
    X = X - X_max
    exp = nd.exp(X)
    partition = nd.sum(exp, axis=1, keepdims=True)
    return exp / partition

def focal_loss(yhat, y, alpha, beta=2.0):
    alpha = alpha.reshape((1, -1))
    alpha_matrix = alpha.broadcast_to(shape=yhat.shape)
    Pt = nd.pick(yhat, y, axis=1, keepdims=True)
    return - nd.pick(alpha_matrix, y, axis=1, keepdims=True) * ((1.0 - Pt)**beta) * nd.log(Pt)

def cross_entropy(yhat, y):
    return - nd.log(nd.pick(yhat, y, axis=1, keepdims=True))

def accuracy(output, label):
    return nd.mean(output.argmax(axis=1)==label).asscalar()

def evaluate_accuracy(data_iter):
    acc = .0
    logloss = .0
    for data, label, angle in data_iter:
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        angle = angle.as_in_context(ctx)
        net_out = net(data)
        angle_out = add_angle(net_out, angle)
        output = softmax(angle_out)
        loss = cross_entropy(output, label)
        logloss += nd.mean(loss).asscalar()
        acc += accuracy(angle_out, label)
    return acc / len(data_iter), logloss / len(data_iter)

def predict(data_iter, net, filename, ctx=[mx.cpu()]):
    outf = open(filename, 'w')
    for data, iden, angle in data_iter:
        data = data.as_in_context(ctx)
        angle = angle.as_in_context(ctx)
        net_out = net(data)
        angle_out = add_angle(net_out, angle)
        prob = softmax(angle_out)
        for i in range(len(iden)):
            # print(prob[i])
            outf.write('%s,%f\n' % (iden[i], prob[i][1].asscalar()))

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

def img_norm(img):
    max_val = nd.max(img)
    min_val = nd.min(img)
    return (img - min_val) / (max_val - min_val)

def angle_norm(angle):
    an = (angle.asscalar() - 30.0) / 16.0
    return an + 1 if 0.0 <= an <= 1.0 else 0

def train(train_data, valid_data, test_data, batch_size):
    epoches = 50
    alpha = nd.array([0.75, 1.0], ctx=ctx)
    for e in range(epoches):
        total_loss = .0
        train_acc = .0
        start = time.time()
        t = 1
        for data, label, angle in train_data:
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            angle = angle.as_in_context(ctx)
            with ag.record():
                net_out = net(data)
                angle_out = add_angle(net_out, angle)
                output = softmax(angle_out)
                loss = focal_loss(output, label, alpha, beta=2.0)
            loss.backward()
            adam(params, vs, sqrs, 0.001, batch_size, t)
            t += 1
            trainer.step(batch_size)
            train_acc += accuracy(output, label)
            total_loss += nd.mean(loss).asscalar()
        test_acc, logloss = evaluate_accuracy(valid_data)
        print("e: %d, train_loss: %f, train_acc: %f, test_acc: %f, logloss: %f, cost_time: %d" % (e, total_loss / len(train_data), \
              train_acc / len(train_data), test_acc, logloss, time.time()- start))
        predict(test_data, net, './predict_result/result.epoch_%d' % e, ctx)

def augment_data(imags, label, angle):
    datas, labels, angles = [], [], []
    n = imags.shape[0]
    for k in range(n):
        imag = imags[k].reshape(shape=(1, imags[k].shape[0], imags[k].shape[1], imags[k].shape[2]))
        for i in range(4):
            trans_band = transform(imag, image.HorizontalFlipAug(.5)).astype('float32')
            datas.append(img_norm(trans_band))
            labels.append(label[k].asscalar())
            angles.append(angle_norm(angle[k]))

        for i in range(9):
            trans_band = transform(imag, image.RandomSizedCropAug((75, 75), .75, (.8, 1.2))).astype('float32')
            datas.append(img_norm(trans_band))
            labels.append(label[k].asscalar())
            angles.append(angle_norm(angle[k]))
        # brightness augmenter
        for i in range(9):
            trans_band = transform(imag, image.BrightnessJitterAug(.1)).astype('float32')
            datas.append(img_norm(trans_band))
            labels.append(label[k].asscalar())
            angles.append(angle_norm(angle[k]))
        # random crop augmenter
        for i in range(9):
            trans_band = resize(transform(imag, image.RandomCropAug((50,50))), 75).astype('float32')
            datas.append(img_norm(trans_band))
            labels.append(label[k].asscalar())
            angles.append(angle_norm(angle[k]))
        # center crop augmenter
        trans_band = resize(transform(imag, image.CenterCropAug((50,50))), 75).astype('float32')
        datas.append(img_norm(trans_band))
        labels.append(label[k].asscalar())
        angles.append(angle_norm(angle[k]))

    ds = nd.concat(*datas, dim=0)
    return ds, labels, angles

def read_src_data(train=True):
    datas, addition, angles = [], [], []
    cnt = 0

    path = 'input/train.json' if train else 'input/test.json'
    with open(path) as f:
        data = json.load(f)
        for img in data:
            angle = img['inc_angle'] if img['inc_angle'] != 'na' else 0
            band_1 = nd.array(img['band_1']).reshape((1, 75, 75, 1))
            band_2 = nd.array(img['band_2']).reshape((1, 75, 75, 1))
            # band_3 = (band_1.astype('float32') + band_2.astype('float32')) / 2
            band = nd.concat(band_1, band_2, dim=3)
            datas.append(band)
            if train:
                addition.append(img['is_iceberg'])
            else:
                addition.append(img['id'])
            angles.append(angle)

    ds = nd.concat(*datas, dim=0)
    return ds, addition, angles

if __name__ == '__main__':
    ds, labels, angles_train = read_src_data()
    test, ids, angles_test = read_src_data(False)
    print("finish load data")

    num = ds.shape[0]
    idx = np.arange(num)
    # np.random.shuffle(idx)
    split = num // 16
    test_idx = idx[:split]
    train_idx = idx[split:]
    print(train_idx.shape)
    print(test_idx.shape)
    sys.stdout.flush()

    train_ds = (
        nd.array(ds.asnumpy()[train_idx]).astype('float32'),
        nd.array(np.array(labels)[train_idx]).astype('float32'),
        nd.array(np.array(angles_train)[train_idx]).astype('float32')
        )
    valid_ds = (
        nd.array(ds.asnumpy()[valid_idx]).astype('float32'),
        nd.array(np.array(labels)[valid_idx]).astype('float32'),
        nd.array(np.array(angles_train)[valid_idx]).astype('float32')
        )
    train_ds_aug, label_train_aug, angle_train_aug = augment_data(train_ds[0], train_ds[1], train_ds[2])
    # print(label_train_aug)
    train_ds = (
        train_ds_aug.astype('float32'),
        nd.array(label_train_aug).astype('float32'),
        nd.array(angle_train_aug).astype('float32')
        )
    valid_ds_aug, label_valid_aug, angle_valid_aug = augment_data(valid_ds[0], valid_ds[1], valid_ds[2])
    valid_ds = (
        valid_ds_aug.astype('float32'),
        nd.array(label_valid_aug).astype('float32'),
        nd.array(angle_valid_aug).astype('float32')
        )

    test_norm = []
    angle_test_norm = []
    for k in range(test.shape[0]):
        imag = test[k].reshape(shape=(1, test[k].shape[0], test[k].shape[1], test[k].shape[2]))
        test_norm.append(img_norm(imag))

    for k in range(angles_test.shape[0]):
        angle_test_norm.append(test_norm(angles_test[k].asscalar()))

    test_ds = (nd.concat(*test_norm, dim=0).astype('float32'), ids, angle_test_norm)

    batch_size = 128
    train_data = DataLoader(train_ds, batch_size, shuffle=True)
    valid_data = DataLoader(valid_ds, batch_size, shuffle=False)
    test_data = TestDataLoader(test_ds, batch_size)

    test_norm = []
    for k in range(test.shape[0]):


    print(len(test_ds[0]))

    train(train_data, valid_data, test_data, batch_size)
