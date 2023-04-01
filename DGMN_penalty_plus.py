from model.network import create_unet_model
import tensorflow as tf
from config import config
import tensorflow_probability as tfp
import numpy as np
from pylab import figure,imshow,show,subplot
from sklearn.metrics import normalized_mutual_info_score


class My_step_my(tf.keras.layers.Layer):
    def __init__(self):
        super(My_step_my, self).__init__()
        self.K = config.ClusterNo
        self.start_Penalty = 0
        self.penalty = 0.05
        self.reg_covar = 1e-7

    def __call__(self, X, Gama, epoch, step, mu_previous, cov_previous, alpha_previous):
        self.mu = mu_previous
        self.cov = cov_previous
        self.alpha = alpha_previous
        # =========================== start m-step =========================
        N, D = tf.shape(X)
        std = tf.math.reduce_std(X, 0)
        mean = tf.math.reduce_mean(X, 0)
        newmu = []
        newcov = []
        newalpha = []
        for k in range(self.K):
            Nk = tf.reduce_sum(Gama[:, k])
            mu_ = tf.squeeze(tf.matmul(tf.transpose(tf.expand_dims(Gama[:, k], -1)), X) / Nk)
            # compute penalty if necessary
            if epoch >= self.start_Penalty:
                shift_board = []
                for p in range(3):
                    shift_board.append(cov_previous[k, p, p] / (std[p] * Nk))
                shift = tf.stack(shift_board, axis=0)
                real_shift = tf.linalg.diag(tf.where(tf.greater(self.mu[k], mean), -1.0, 1.0)) * shift * self.penalty
                newmu.append(mu_ + tf.linalg.diag_part(real_shift))
            else:
                newmu.append(mu_)
            newalpha.append(Nk/tf.cast(N, tf.float32))
        mu = tf.stack(newmu, axis=0)
        for k in range(self.K):
            # k is the number of classes
            diff = X - mu[k]  # note this mu must be the previous one in last iteration
            cov_k = tf.matmul(tf.transpose(diff),
                              tf.multiply(diff, tf.expand_dims(Gama[:, k], -1) / Nk))  # + tf.eye(3) * self.reg_covar
            newcov.append(cov_k)
        cov = tf.stack(newcov, axis=0)
        alpha = tf.stack(newalpha, axis=0)
        # ===========================  m-step finished =========================
        # conduct inner-E-step to get log-prob-norm
        prob = []
        for k in range(self.K):
            jitter = 1e-5
            fix_cov_k_value = tf.linalg.det(cov[k])
            while fix_cov_k_value <= 0:
                fix_cov_k = cov[k] + tf.linalg.tensor_diag([jitter, jitter, jitter])
                fix_cov_k_value = tf.linalg.det(fix_cov_k)
                jitter *= 10
            if tf.linalg.det(cov[k]) >0:
                probility = tfp.distributions.MultivariateNormalFullCovariance(mu[k], cov[k]).prob(X)
            else:
                print("fix cov, the jitter is ", jitter)
                probility = tfp.distributions.MultivariateNormalFullCovariance(mu[k], fix_cov_k).prob(X)
            prob.append(probility)
        pi_prob_stack = alpha * tf.squeeze(tf.stack(prob, axis=-1))
        # compute loglikelihood
        log_prob = tf.math.log(pi_prob_stack + 1e-5)
        prob_sumexp = tf.reduce_sum(tf.exp(log_prob), axis=1)
        loglikelihood = tf.math.log(prob_sumexp)
        # log(M/N) = logM- logN, loglikelihood actually is the sum of the pi[k] * prob
        log_gamma = log_prob - loglikelihood[:, tf.newaxis]
        new_gamma = tf.exp(log_gamma)
        # build the optimum function
        # constraint = tf.reduce_sum(tf.abs(cov[0,...]))
        if epoch >= self.start_Penalty:
            penalty = self.penalty * tf.reduce_sum(tf.abs(mu - mean) / std)
            loss = tf.negative(tf.reduce_mean(loglikelihood) - penalty) #- constraint *0.1
        else:
            loss = tf.negative(tf.reduce_mean(loglikelihood)) #- constraint *0.1
        return loss, new_gamma, mu, cov, alpha


class Build_DGMN(tf.keras.Model):
    def __init__(self):
        super(Build_DGMN, self).__init__()
        self.E_step = create_unet_model()
        self.M_step = My_step_my()
        self.D = 3

    def __data_normalize__(self, X, method='simple'):
        x_nor = []
        if method == "simple":
            for d in range(self.D):
                max_ = tf.reduce_max(X[..., d])
                min_ = tf.reduce_min(X[..., d])
                x_nor.append((X[..., d] - min_)/(max_ - min_))
            x_nor = tf.stack(x_nor, axis=-1)
        elif method == "advanced":
            for d in range(self.D):
                std_ = tf.math.reduce_std(X[..., d])
                mean_ = tf.reduce_mean(X[..., d])
                img_temp_normalized = (X[..., d] - mean_)/std_
                x_nor.append((img_temp_normalized - tf.reduce_min(img_temp_normalized))/
                             (tf.reduce_max(img_temp_normalized) - tf.reduce_min(img_temp_normalized)))
            x_nor = tf.stack(x_nor, axis=-1)
        else:
            print("no data normalization implemented !")
        return x_nor

    def __call__(self, X, training, epoch=None, step=None, mu_previous=None,
                 cov_previous=None, alpha_previous=None, return_mixpar=False):
        X_normal = self.__data_normalize__(X, method='simple')
        X_reshape = tf.reshape(X_normal, (-1, self.D))
        if training:
            gamma = self.E_step(X_normal, training=training)
            gamma_reshape = tf.reshape(gamma, (-1, config.ClusterNo))
            loss, new_gamma, mu, cov, alpha = self.M_step(X_reshape, gamma_reshape, epoch,
                                                          step, mu_previous, cov_previous, alpha_previous)
            gmm_map = np.argmax(new_gamma.numpy().squeeze().reshape(config.img_size,config.img_size,config.ClusterNo), axis=-1)
            return loss, mu, cov, alpha, gmm_map
        else:
            pred = self.E_step(X_normal, training=False)
            gamma_reshape = tf.reshape(pred, (-1, config.ClusterNo))
            if return_mixpar:
                loss, new_gamma, mu, cov, alpha = self.M_step(X_reshape, gamma_reshape, epoch,
                                                              step, mu_previous, cov_previous, alpha_previous)
                return pred, new_gamma, mu, cov, alpha, X_normal
            else:
                return pred

