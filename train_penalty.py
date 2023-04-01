import tensorflow as tf
from utils.data_loader_cells import Dataset
from model.DGMN_penalty_plus import Build_DGMN
import numpy as np
from config import config
import cv2
import os
import matplotlib.pyplot as plt
import csv
from utils.utils import compute_dice
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import f1_score, precision_score,recall_score

os.environ['CUDA_VISIBLE_DEVICES']='0'

if __name__ == '__main__':
    if not os.path.exists(config.log_path):
        os.mkdir(config.log_path)
    if not os.path.exists(config.valid_visualize_path):
        os.mkdir(config.valid_visualize_path)
    if not os.path.exists(config.save_model_dir):
        os.mkdir(config.save_model_dir)
    print("dir path check")
    gpus = tf.config.list_physical_devices("GPU")
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    configs = tf.config.experimental.set_memory_growth(physical_devices[0], True)
    np.set_printoptions(suppress=True)

    data = Dataset(config.data_root_path, config.img_size)
    steps_per_epoch = tf.math.ceil(data.length_train / config.batch_size)

    # define opt
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-5,
                                                                 decay_steps=steps_per_epoch * config.learning_rate_decay_epochs,
                                                                 decay_rate=0.95)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    network = Build_DGMN()
    loss_metric = tf.metrics.Mean()
    best_dice = 0
    save_dice = 0.74

    if 1:
        network.load_weights(filepath="./log/DGMN_lasso/weights/epoch-76-0.743.ckpt")
        print("load successful")

    def train_step(batch_X, epoch, step, mu_pre, cov_pre, alpha_pre):
        with tf.GradientTape() as tape:
            losses, mu, cov, alpha, newgamma = network(batch_X, training=True, epoch=epoch, step=step, mu_previous=mu_pre,
                                             cov_previous=cov_pre, alpha_previous=alpha_pre)
            # lossL2 = tf.reduce_sum([tf.nn.l2_loss(v) for v in network.trainable_variables if 'kernel' in v.name]) * 0.0005
            final = losses
        gradients = tape.gradient(target=final, sources=network.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, network.trainable_variables))
        loss_metric.update_state(values=losses)
        return mu, cov, alpha, newgamma

    @tf.function
    def data_augmentation(images):
        images = tf.image.random_flip_up_down(images)
        images = tf.image.random_brightness(images, 0.15)
        images = tf.image.random_saturation(images, lower=0.5, upper=1.5)
        images = tf.image.random_hue(images, 0.12)
        images = tf.image.random_flip_left_right(images)
        return images

    loss_minimum = 100
    sigma_previous = tf.random.uniform(shape=(3, config.ClusterNo))
    iter = 0
    iter_list = []
    train_loss_list = []
    mu_previous = np.random.rand(config.ClusterNo, 3)
    cov_previous = np.repeat(np.expand_dims(np.eye(3),0), config.ClusterNo, axis=0) * 0.1
    alpha_previous = np.array([1.0/config.ClusterNo] * config.ClusterNo)

    for epoch in range(config.epochs):
        loss_record = []
        for step, batch_data in enumerate(data.train_data):
            fids, images = batch_data
            images = data_augmentation(images)
            mu_current, cov_current, alpha_current, _ = train_step(images, epoch, step,
                                                                mu_previous, cov_previous, alpha_previous)
            mu_previous = mu_current
            cov_previous = cov_current
            alpha_previous = alpha_current
            if step % config.train_print_step == 0:
                print("Epoch {}/{}, step: {}/{}, loss: {}".format(epoch, config.epochs, step,
                                                                  steps_per_epoch, loss_metric.result()))
                train_loss_list.append(loss_metric.result())
                iter_list.append(iter)
                iter += 1
            loss_record.append(loss_metric.result())
        loss_metric.reset_states()
        loss_present_epoch = np.mean(loss_record)

        plt.figure()
        plt.plot(iter_list, train_loss_list)
        plt.xlabel(u'iters')
        plt.ylabel(u'loss')
        plt.savefig("train_results_loss.png")

        # start valid process
        if epoch % config.valid_freq == 0:
            mdice = []
            mrec = []
            mprec = []
            if not os.path.exists(config.valid_visualize_path + str(epoch)):
                os.mkdir(config.valid_visualize_path + str(epoch))
            for step, batch_data in enumerate(data.valid_data):
                img, fid = batch_data
                real_fid = fid.numpy().decode('gbk').split('/')[-1]
                pd_gamma = network(img, training=False)
                segment_results = np.argmax(pd_gamma, axis=3).squeeze()
                pred_ = segment_results.copy()
                for i in range(config.ClusterNo):
                    segment_results[segment_results==i] = 100 * i
                img_numpy = (img.numpy().squeeze()).astype(np.uint8)
                # cv2.imwrite(config.valid_visualize_path+'{}/{}.png'.format(epoch, step), img_numpy)
                cv2.imwrite(config.valid_visualize_path+'{}/{}'.format(epoch, real_fid),
                            segment_results.astype(np.uint8))
                # evaluate metrics
                if config.val_dice:

                    gt = cv2.imread('./data/cells/valid/masks/' + real_fid, 0)
                    gt = cv2.resize(gt, (config.img_size, config.img_size))
                    _, thresh = cv2.threshold(gt, 125, 255, cv2.THRESH_BINARY)
                    gt_ = thresh / 255
                    # print(np.unique(pred_), np.unique(gt_))
                    assert len(np.unique(gt_)) == 2, 'label error'
                    pred_onehot = to_categorical(pred_, config.ClusterNo)
                    dice0 = compute_dice(pred_onehot[..., 0], gt_)
                    dice1 = compute_dice(pred_onehot[..., 1], gt_)
                    dice2 = compute_dice(pred_onehot[..., -1], gt_)
                    dicemax = np.max(np.array((dice0, dice1, dice2)))
                    idx = np.argmax(np.array((dice0, dice1, dice2)))
                    precision = precision_score(gt_.reshape(-1), pred_onehot[..., idx].reshape(-1))
                    recall = recall_score(gt_.reshape(-1), pred_onehot[..., idx].reshape(-1))
                    mdice.append(dicemax)
                    mprec.append(precision)
                    mrec.append(recall)
            if config.val_dice:
                print("valid-----Epoch: {}/{}, dice: {}, precision:{}, recall: {}, the best is {}".format(epoch, config.epochs,
                                                                                              np.mean(mdice),
                                                                                              np.mean(mprec),
                                                                                              np.mean(mrec), best_dice))
        epoch_dice = np.mean(mdice)
        if config.val_dice:
            if epoch_dice > best_dice:
                best_dice = epoch_dice
            if epoch_dice > save_dice:
                network.save_weights(filepath=config.save_model_dir + "epoch-{}-{}.ckpt".format(epoch, np.round(epoch_dice, 3)), save_format="tf")
        else:
            network.save_weights(filepath=config.save_model_dir + "epoch-{}.ckpt".format(epoch), save_format="tf")
