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

class Net_vgg10(gluon.nn.Block):
    def __init__(self, **kwargs):
        super(Net_vgg10, self).__init__(**kwargs)
        with self.name_scope():
            self.net = gluon.nn.Sequential()
            self.net.add(
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

                gluon.nn.Dense(2),
            )

    def forward(self, x):
        return self.net(x)


def train(net_vgg, train_data, valid_data, test_data, batch_size, num_epochs, lr, ctx):
    trainer = gluon.Trainer(
        net_vgg.collect_params(), 'adam', {'learning_rate': lr,})

    max_entropy_loss = gluon.loss.SoftmaxCrossEntropyLoss()
    prev_time = datetime.datetime.now()
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_acc = 0.0
        batch = 0
        for data, label in train_data:
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            with autograd.record():
                output = net_vgg(data)
                loss = max_entropy_loss(output, label)
            loss.backward()
            trainer.step(batch_size)
            train_loss += nd.mean(loss).asscalar()
            train_acc += utils.accuracy(output, label)
            batch += 1

        cur_time = datetime.datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)

        valid_acc, test_loss = utils.evaluate_accuracy(valid_data, net_vgg, ctx)
        epoch_str = ("Epoch %d. Loss: %f, Train acc %f, Valid acc %f, Test loss: %f "
                     % (epoch, train_loss / len(train_data),
                        train_acc / len(train_data), valid_acc, test_loss))
        prev_time = cur_time
        print(epoch_str + time_str + ', lr ' + str(trainer.learning_rate))
        sys.stdout.flush()
        net_vgg.save_params('./model_out/vggnet_epoch_%d' % epoch)
        utils.predict(test_data, net_vgg, './predict_result/result.epoch_%d' % epoch, ctx)


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

def augment_data(imags, label):
    datas, labels = [], []
    n = imags.shape[0]
    for k in range(n):
        imag = imags[k].reshape(shape=(1, imags[k].shape[0], imags[k].shape[1], imags[k].shape[2]))
        for i in range(4):
            trans_band = transform(imag, image.HorizontalFlipAug(.5)).astype('float32')
            datas.append(img_norm(trans_band))
            labels.append(label[k].asscalar())

        for i in range(9):
            trans_band = transform(imag, image.RandomSizedCropAug((75, 75), .75, (.8, 1.2))).astype('float32')
            datas.append(img_norm(trans_band))
            labels.append(label[k].asscalar())
        # brightness augmenter
        for i in range(9):
            trans_band = transform(imag, image.BrightnessJitterAug(.1)).astype('float32')
            datas.append(img_norm(trans_band))
            labels.append(label[k].asscalar())
        # random crop augmenter
        for i in range(9):
            trans_band = resize(transform(imag, image.RandomCropAug((50,50))), 75).astype('float32')
            datas.append(img_norm(trans_band))
            labels.append(label[k].asscalar())
        # center crop augmenter
        trans_band = resize(transform(imag, image.CenterCropAug((50,50))), 75).astype('float32')
        datas.append(img_norm(trans_band))
        labels.append(label[k].asscalar())
    ds = nd.concat(*datas, dim=0)
    return ds, labels

def read_src_data(train=True):
    datas, addition = [], []
    cnt = 0

    path = 'input/train.json' if train else 'input/test.json'
    with open(path) as f:
        data = json.load(f)
        for img in data:
            band_1 = nd.array(img['band_1']).reshape((1, 75, 75, 1))
            band_2 = nd.array(img['band_2']).reshape((1, 75, 75, 1))
            # band_3 = (band_1.astype('float32') + band_2.astype('float32')) / 2
            band = nd.concat(band_1, band_2, dim=3)
            datas.append(band)
            if train:
                addition.append(img['is_iceberg'])
            else:
                addition.append(img['id'])

    ds = nd.concat(*datas, dim=0)
    return ds, addition

if __name__ == '__main__':
    # gen_2channel_img()
    ds, labels = read_src_data()
    test, ids = read_src_data(False)
    print('testset size: %d' % test.shape[0])
    print("finish load data")

    num = ds.shape[0]
    idx = np.arange(num)
    np.random.shuffle(idx)
    split = num // 16
    valid_idx = idx[:split]
    train_idx = idx[split:]
    print(train_idx.shape)
    print(valid_idx.shape)
    sys.stdout.flush()

    max_val = 40.
    min_val = -50.

    train_ds = (
        nd.array(ds.asnumpy()[train_idx]).astype('float32'),
        nd.array(np.array(labels)[train_idx]).astype('float32')
        )
    train_ds_aug, label_train_aug = augment_data(train_ds[0], train_ds[1])
    train_ds = (
        # (nd.clip(train_ds_aug, -50, 40) - min_val) / (max_val - min_val),
        train_ds_aug,
        nd.array(label_train_aug).astype('float32')
        )

    valid_ds = (
        nd.array(ds.asnumpy()[valid_idx]).astype('float32'),
        nd.array(np.array(labels)[valid_idx]).astype('float32')
        )
    valid_ds_aug, label_valid_aug = augment_data(valid_ds[0], valid_ds[1])
    valid_ds = (
        # (nd.clip(valid_ds_aug, -50, 40) - min_val) / (max_val - min_val),
        valid_ds_aug,
        nd.array(label_valid_aug).astype('float32')
        )
    
    # test_ds = ((nd.clip(test, -50, 40) - min_val) / (max_val - min_val), ids)
    test_norm = []

    for k in range(test.shape[0]):
        imag = test[k].reshape(shape=(1, test[k].shape[0], test[k].shape[1], test[k].shape[2]))
        test_norm.append(img_norm(imag))

    test_ds = (nd.concat(*test_norm, dim=0).astype('float32'), ids)

    print("max/min train: %f\t%f" % (nd.max(train_ds[0]).asscalar(), nd.min(train_ds[0]).asscalar()))
    print("max/min valid: %f\t%f" % (nd.max(valid_ds[0]).asscalar(), nd.min(valid_ds[0]).asscalar())) 
    print("max/min test: %f\t%f" % (nd.max(test_ds[0]).asscalar(), nd.min(test_ds[0]).asscalar()))

    print("finish gen train/valid dataset")

    batch_size = 128
    train_data = utils.DataLoader(train_ds, batch_size, shuffle=True)
    valid_data = utils.DataLoader(valid_ds, batch_size, shuffle=False)
    test_data = utils.TestDataLoader(test_ds, batch_size)

    ctx = utils.try_gpu()
    num_epochs = 100
    
    learning_rate = .001

    net = Net_vgg10()
    net.initialize(init=init.Xavier(), ctx=ctx)
    # net.hybridize()
    print("Start training on ", ctx)
    sys.stdout.flush()
    train(net, train_data, valid_data, test_data,
            batch_size, num_epochs, learning_rate, ctx)
