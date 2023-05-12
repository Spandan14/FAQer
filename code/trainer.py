import os
import pickle
import time

import tensorflow as tf
import numpy as np

import config
from data_utils import get_loader, eta, user_friendly_time, progress_bar, time_since
from model import Seq2Seq

class Trainer(object):
    def __init__(self, args):
        # load dict and embeddings
        with open(config.embedding, "rb") as f:
            embedding = pickle.load(f)
            embedding = tf.Variable(embedding, dtype=tf.float32)
            embedding = tf.Variable(embedding, dtype=tf.float32)

        with open(config.word2idx_file, "rb") as f:
            word2idx = pickle.load(f)
        
        # train, dev loader
        print("Loading training data...")
        self.train_loader, _ = get_loader(config.train_src_file,
                                       config.train_trg_file,
                                       word2idx,
                                       use_tag=True,
                                       batch_size=config.batch_size,
                                       debug=config.debug)
        self.dev_loader, _ = get_loader(config.dev_src_file,
                                     config.dev_trg_file,
                                     word2idx,
                                     use_tag=True,
                                     batch_size=128,
                                     debug=config.debug)
        self.train_dataset, self.tr_sz = get_loader(config.train_src_file,
        self.train_loader, _ = get_loader(config.train_src_file,
                                       config.train_trg_file,
                                       word2idx,
                                       use_tag=True,
                                       batch_size=config.batch_size,
                                       debug=config.debug)
        self.dev_loader, _ = get_loader(config.dev_src_file,
                                     config.dev_trg_file,
                                     word2idx,
                                     use_tag=True,
                                     batch_size=128,
                                     debug=config.debug)
        self.train_dataset, self.tr_sz = get_loader(config.train_src_file,
                                       config.train_trg_file,
                                       word2idx,
                                       use_tag=True,
                                       batch_size=config.batch_size,
                                       debug=config.debug)
        self.dev_dataset, self.dv_sz = get_loader(config.dev_src_file,
        self.dev_dataset, self.dv_sz = get_loader(config.dev_src_file,
                                     config.dev_trg_file,
                                     word2idx,
                                     use_tag=True,
                                     batch_size=128,
                                     debug=config.debug)
        
        train_dir = os.path.join("./save", "seq2seq")
        self.model_dir = os.path.join(
            train_dir, "train_%d" % int(time.strftime("%m%d%H%M%S")))
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        self.model = Seq2Seq(embedding)

        if len(args.model_path) > 0:
            print(f"Loading checkpoint from {args.model_path}")
            self.model = tf.keras.models.load_model(args.model_path)   

        self.lr = config.lr
        self.optimizer = tf.keras.optimizers.SGD(self.lr, momentum=0.8)

        # self.loss = tf.keras.losses.SparseCategoricalCrossentropy(ignore_class=0)
        self.loss = tf.keras.losses.BinaryCrossentropy()
    
    def save_model(self, loss, epoch, targets):
        loss = round(loss, 2)
        # model_save_path = os.path.join(self.model_dir, str(epoch) + "_" + str(loss))
        print(loss, targets)
        # self.model.save(model_save_path)

    
    def train(self):
        batch_num = len(self.train_loader)
        best_loss = 1e10
        for epoch in range(1, config.num_epochs + 1):
            # print("epoch {epoch}/{config.num_epochs} :", end="\r")
            print(epoch/config.num_epochs)
            start = time.time()

            if epoch >= 8 and epoch % 2 == 0:
                self.lr *= 0.5
                self.optimizer.learning_rate.assign(self.lr)
            
            for batch_idx, train_data in enumerate(self.train_loader, start=1):
                batch_loss, targets = self.training_step(train_data)
                # print(f"TRAIN | {batch_idx}/{batch_num} {progress_bar(batch_idx, batch_num)} | ETA: {eta(start, batch_idx, batch_num)} | loss: {round(batch_loss, 4)}", end="\r")
                print(batch_idx/batch_num, batch_loss)

            val_loss = self.evaluate()
            if val_loss <= best_loss:
                best_loss = val_loss
                self.save_model(val_loss, epoch, targets)
            
            # print(f"EPOCH | {epoch} | runtime: {user_friendly_time(time_since(start))} | final loss: {round(batch_loss, 4)} | val loss: {round(val_loss, 4)}")
            print(batch_loss, val_loss)

    def training_step(self, train_data):
        # src_seq, ext_src_seq, trg_seq, ext_trg_seq, tag_seq, _ = train_data

        # print(len(train_data[0]), self.tr_sz)

        src_seq, trg_seq = tf.split(train_data, [self.tr_sz[0], self.tr_sz[1]], axis=1)
        ext_src_seq, tag_seq = tf.split(train_data, [self.tr_sz[0], self.tr_sz[1]], axis=1)
        _, ext_trg_seq = tf.split(train_data, [self.tr_sz[0], self.tr_sz[1]], axis=1)
        # src_seq, ext_src_seq, trg_seq, ext_trg_seq, tag_seq, _ = train_data

        # print(len(train_data[0]), self.tr_sz)

        src_seq, trg_seq = tf.split(train_data, [self.tr_sz[0], self.tr_sz[1]], axis=1)
        ext_src_seq, tag_seq = tf.split(train_data, [self.tr_sz[0], self.tr_sz[1]], axis=1)
        _, ext_trg_seq = tf.split(train_data, [self.tr_sz[0], self.tr_sz[1]], axis=1)

        eos_trg = ext_trg_seq[:, 1:] # ext_trg_seq[:, 1:]
        eos_trg = ext_trg_seq[:, 1:] # ext_trg_seq[:, 1:]

        with tf.GradientTape() as tape:
            logits = tf.cast(self.model(src_seq, tag_seq, ext_src_seq, trg_seq), dtype=tf.float32)
        with tf.GradientTape() as tape:
            logits = tf.cast(self.model(src_seq, tag_seq, ext_src_seq, trg_seq), dtype=tf.float32)

            batch_size = logits.shape[0]
            nsteps = logits.shape[1]

            # print(logits.shape)

            preds = tf.cast(tf.reshape(logits, (batch_size * nsteps, -1)), dtype=tf.float32)
            # print(preds[:, 0].shape)
            targets = tf.cast(tf.reshape(eos_trg, -1), dtype=tf.float32)

            # print(targets.shape)

            loss = tf.cast(self.loss(preds[:, 0], targets), dtype=tf.float32)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss, preds

    def test_step(self, test_data):
        # src_seq, ext_src_seq, trg_seq, ext_trg_seq, tag_seq, _ = test_data

        src_seq, trg_seq = tf.split(test_data, [self.dv_sz[0], self.dv_sz[1]], axis=1)
        ext_src_seq, tag_seq = tf.split(test_data, [self.dv_sz[0], self.dv_sz[1]], axis=1)
        _, ext_trg_seq = tf.split(test_data, [self.dv_sz[0], self.dv_sz[1]], axis=1)

        eos_trg = ext_trg_seq[:, 1:]

        logits = tf.cast(self.model(src_seq, tag_seq, ext_src_seq, trg_seq), dtype=tf.float32)
        logits = tf.cast(self.model(src_seq, tag_seq, ext_src_seq, trg_seq), dtype=tf.float32)

        batch_size = logits.shape[0]
        nsteps = logits.shape[1]

        preds = tf.cast(tf.reshape(logits, (batch_size * nsteps, -1)), dtype=tf.float32)
        targets = tf.cast(tf.reshape(eos_trg, -1), dtype=tf.float32)
        loss = tf.cast(self.loss(preds[:, 0], targets), dtype=tf.float32)

        return loss


    def evaluate(self):
        batch_num = len(self.dev_loader)
        test_losses = []
        for batch_idx, test_data in enumerate(self.dev_loader, start=1):
            batch_loss = self.test_step(test_data)
            test_losses.append(batch_loss)
            # print(f"TEST  | {batch_idx}/{batch_num} {progress_bar(batch_idx, batch_num)} | loss: {round(batch_loss, 4)}", end="\r")
            print(batch_idx/batch_num, batch_loss)
        
        return np.mean(test_losses)
