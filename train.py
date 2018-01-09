import sys, datetime, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl

from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon
from mxnet import image
from mxnet import init
from mxnet import nd
from mxnet.gluon.data import vision
import utils

import imageio

def net_vgg7(ctx):
    net_vgg = gluon.nn.Sequential()
    with net_vgg.name_scope():
        net_vgg.add(
            gluon.nn.Conv2D(channels=32, kernel_size=3, padding=1),
            gluon.nn.BatchNorm(axis=1),
            gluon.nn.Activation(activation='relu'),
            gluon.nn.Dropout(0.5),
            gluon.nn.Conv2D(channels=32, kernel_size=3, padding=1),
            gluon.nn.BatchNorm(axis=1),
            gluon.nn.Activation(activation='relu'),
            gluon.nn.MaxPool2D(pool_size=2),
            gluon.nn.Dropout(0.5),

            gluon.nn.Conv2D(channels=64, kernel_size=3, padding=1),
            gluon.nn.BatchNorm(axis=1),
            gluon.nn.Activation(activation='relu'),
            gluon.nn.Dropout(0.5),
            gluon.nn.Conv2D(channels=64, kernel_size=3, padding=1),
            gluon.nn.BatchNorm(axis=1),
            gluon.nn.Activation(activation='relu'),
            gluon.nn.MaxPool2D(pool_size=2),
            gluon.nn.Dropout(0.5),

            gluon.nn.Fletten(),
            gluon.nn.Dense(256),
            gluon.nn.BatchNorm(axis=1),
            gluon.nn.Activation(activation='relu'),
            gluon.nn.Dropout(0.5),
            gluon.nn.Dense(256),
            gluon.nn.BatchNorm(axis=1),
            gluon.nn.Activation(activation='relu'),
            gluon.nn.Dropout(0.5),
            gluon.nn.Dense(1),
            gluon.nn.Activation(activation='sigmoid'),
        )
    net.initialize(ctx=ctx, init=init.Xavier())
    return net_vgg


def net_vgg10(ctx):
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(
            gluon.nn.Conv2D(channels=32, kernel_size=3, padding=1),
            gluon.nn.BatchNorm(axis=1),
            gluon.nn.Activation(activation='relu'),
            gluon.nn.Dropout(0.2),
            gluon.nn.Conv2D(channels=32, kernel_size=3, padding=1),
            gluon.nn.BatchNorm(axis=1),
            gluon.nn.Activation(activation='relu'),
            gluon.nn.MaxPool2D(pool_size=3, strides=2),
            gluon.nn.Dropout(0.5),

            gluon.nn.Conv2D(channels=64, kernel_size=3, padding=1),
            gluon.nn.BatchNorm(axis=1),
            gluon.nn.Activation(activation='relu'),
            gluon.nn.Dropout(0.2),
            gluon.nn.Conv2D(channels=64, kernel_size=3, padding=1),
            gluon.nn.BatchNorm(axis=1),
            gluon.nn.Activation(activation='relu'),
            gluon.nn.MaxPool2D(pool_size=3, strides=2),
            gluon.nn.Dropout(0.5),

            gluon.nn.Conv2D(channels=128, kernel_size=3, padding=1),
            gluon.nn.BatchNorm(axis=1),
            gluon.nn.Activation(activation='relu'),
            gluon.nn.Dropout(0.2),
            gluon.nn.Conv2D(channels=128, kernel_size=3, padding=1),
            gluon.nn.BatchNorm(axis=1),
            gluon.nn.Activation(activation='relu'),
            gluon.nn.MaxPool2D(pool_size=2),
            gluon.nn.Dropout(0.5),

            gluon.nn.Conv2D(channels=128, kernel_size=3, padding=1),
            gluon.nn.BatchNorm(axis=1),
            gluon.nn.Activation(activation='relu'),
            gluon.nn.Dropout(0.2),
            gluon.nn.Conv2D(channels=128, kernel_size=3, padding=1),
            gluon.nn.BatchNorm(axis=1),
            gluon.nn.Activation(activation='relu'),
            gluon.nn.MaxPool2D(pool_size=3, strides=2),
            gluon.nn.Dropout(0.5),

            gluon.nn.Flatten(),

            gluon.nn.Dense(256),
            gluon.nn.BatchNorm(axis=1),
            gluon.nn.Activation(activation='relu'),
            gluon.nn.Dropout(0.5),

            gluon.nn.Dense(256),
            gluon.nn.BatchNorm(axis=1),
            gluon.nn.Activation(activation='relu'),
            gluon.nn.Dropout(0.5),

            gluon.nn.Dense(1),
            gluon.nn.Activation(activation='sigmoid')
        )
    net.initialize(ctx=ctx, init=init.Xavier())
    return net


def train(net_vgg, train_data, valid_data, batch_size, num_epochs, lr, ctx):
    trainer = gluon.Trainer(
        net_vgg.collect_params(), 'adam', {'learning_rate': lr,})
    logloss = gluon.loss.LogisticLoss()
    prev_time = datetime.datetime.now()
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_acc = 0.0
        batch = 0
        for data, label in train_data:


            label = label.as_in_context(ctx)
            with autograd.record():
                output = net_vgg(data.as_in_context(ctx))
                loss = logloss(output, label)
            loss.backward()
            trainer.step(batch_size)
            train_loss += nd.mean(loss).asscalar()
            train_acc += utils.accuracy(output, label)
            batch += 1

            if batch % 10 == 0:
                batch50_str = ("Batch %d. Loss: %f, Train acc %f, "
                     % (batch, train_loss / len(train_data),
                        train_acc / len(train_data)))
                print(batch50_str)
                sys.stdout.flush()

        cur_time = datetime.datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)

        valid_acc = utils.evaluate_accuracy(valid_data, net_vgg, ctx)
        epoch_str = ("Epoch %d. Loss: %f, Train acc %f, Valid acc %f, "
                     % (epoch, train_loss / len(train_data),
                        train_acc / len(train_data), valid_acc))
        prev_time = cur_time
        print(epoch_str + time_str + ', lr ' + str(trainer.learning_rate))
        sys.stdout.flush()


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

if __name__ == '__main__':
    # gen_2channel_img()
    datas, labels = [], []
    cnt = 0
    with open('./input/train.json') as f:
        data = json.load(f)
        for img in data:
            name = img['id']
            label = -1 if img['is_iceberg'] == 0 else 1
            band_1 = nd.array(img['band_1']).reshape((1, 75, 75, 1))
            band_2 = nd.array(img['band_2']).reshape((1, 75, 75, 1))
            band = nd.concat(band_1, band_2, dim=3)
            print(type(label), label)
            exit(0)

            # source img
            # datas.append(band)
            # labels.append(label)
            # horizon flip augmenter
            for i in range(4):
                trans_band = transform(band, image.HorizontalFlipAug(.5))
                datas.append(trans_band)
                labels.append(label)

            for i in range(9):
                trans_band = transform(band, image.RandomSizedCropAug((75, 75), .75, (.8, 1.2)))
                datas.append(trans_band)
                labels.append(label)
            # brightness augmenter
            for i in range(9):
                trans_band = transform(band, image.BrightnessJitterAug(.3))
                datas.append(trans_band)
                labels.append(label)
            # random crop augmenter
            for i in range(9):
                trans_band = resize(transform(band, image.RandomCropAug((50,50))), 75)
                datas.append(trans_band)
                labels.append(label)
            # center crop augmenter
            trans_band = resize(transform(band, image.CenterCropAug((50,50))), 75)
            datas.append(trans_band)
            labels.append(label)
            cnt += 1
            if cnt % 10 == 0:
                print("%d source image get done." % cnt)
                sys.stdout.flush()
    ds = nd.concat(*datas, dim=0)
    print("finish load data")

    max_val = nd.max(ds.astype('float32'))
    min_val = nd.min(ds.astype('float32'))
    ds = (ds.astype('float32') - min_val) / (max_val - min_val)
    print("max val after normalizing:", nd.max(ds.astype('float32')))
    print("min val after normalizing:", nd.min(ds.astype('float32')))

    num = ds.shape[0]
    idx = np.arange(num)
    np.random.shuffle(idx)
    split = num // 4
    test_idx = idx[:split]
    train_idx = idx[split:]
    print(train_idx.shape)
    print(test_idx.shape)
    sys.stdout.flush()

    train_ds = (
        nd.array(ds.asnumpy()[train_idx]),
        nd.array(np.array(labels)[train_idx])
        )
    test_ds = (
        nd.array(ds.asnumpy()[test_idx]),
        nd.array(np.array(labels)[test_idx])
        )
    batch_size = 128
    train_data = utils.DataLoader(train_ds, batch_size, shuffle=True)
    test_data = utils.DataLoader(test_ds, batch_size, shuffle=False)

    print("finish gen train/test dataset")

    ctx = utils.try_gpu()
    num_epochs = 1
    learning_rate = .001

    net = net_vgg10(ctx)
    net.hybridize()
    print("Start training on ", ctx)
    sys.stdout.flush()
    train(net, train_data, test_data, batch_size, num_epochs, learning_rate, ctx)
