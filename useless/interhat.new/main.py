#! /bin/usr/python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

from data_loader import DataLoader
import numpy as np
from const import Constant
from sklearn.metrics import roc_auc_score

from model import  InterprecsysBase
from utils import create_folder_tree, evaluate_metrics, build_msg

flags = tf.app.flags

# ID
flags.DEFINE_string('trial_id', '001', 'The ID of the current run.')

# Run time
flags.DEFINE_integer('epoch', 300, 'Number of Epochs.')
flags.DEFINE_integer('batch_size', 64, 'Number of training instance per batch.')
flags.DEFINE_string('dataset', 'example', 'Name of the dataset.')
flags.DEFINE_integer('num_iter_per_save', 100, 'Number of iterations per save.')

# Optimization
flags.DEFINE_float('learning_rate', 0.001, 'Learning Rate.')
flags.DEFINE_float('regularization_weight', 0.01, 'The weight of L2-regularization.')

# Parameter Space
flags.DEFINE_integer('embedding_size', 8, 'Hidden Embedding Size.')

# Hyper-param
flags.DEFINE_float('dropout_rate', 0.1, 'The dropout rate of Transformer model.')
flags.DEFINE_integer('highest_order', 4, 'The highest order of cross features.')
flags.DEFINE_float('temperature', 0.8, 'Temperature value of attentions.')

# Structure & Configure
flags.DEFINE_integer('random_seed', 2018, 'Random Seed.')
flags.DEFINE_integer('num_block', 2, 'Number of blocks of Multi-head Attention.')
flags.DEFINE_integer('num_head', 8, 'Number of heads of Multi-head Attention.')
flags.DEFINE_integer('attention_size', 8, 'Number of hidden units in Multi-head Attention.')
flags.DEFINE_integer('attention_size_last', 64, 'Size of pooling filter.')
flags.DEFINE_string('pred_layers', "512,128,32,2", "Sizes of the last layers.")

FLAGS = flags.FLAGS


def run_model(data_loader,
              model,
              epochs=None):
    """Run model (fit/predict)"""

    # configs
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8

    # set dir for runtime log
    log_dir = os.path.join(Constant.LOG_DIR, FLAGS.dataset, "train_" + FLAGS.trial_id)

    # create session
    sess = tf.Session(config=config)

    print("\n====== ID:{} =======\n".format(FLAGS.trial_id))

    # training
    with sess.as_default(), \
        open("./performance/" + FLAGS.dataset + "."
             + FLAGS.trial_id + ".pref", "w") as performance_writer:

        # initializations
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            data_loader.has_next = True
            while data_loader.has_next:

                # get batch, removed batch_value
                batch_ind, batch_label = data_loader.generate_train_batch_ivl()
                batch_label = batch_label.squeeze()

                # run training operation
                op, loss, log_loss, reg_loss, predictions, logits = sess.run(
                    fetches=[
                        model.train_op, model.loss,
                        model.log_loss, model.reg_loss, model.pred,
                        model.logits
                    ],
                    feed_dict={
                        model.X_ind: batch_ind,
                        model.label: batch_label,
                        model.is_training: True
                    }
                )

                logits = logits[:, 1]
                # print(logits.shape)

                # print results and write to file
                if sess.run(model.global_step) \
                        % 10 == 0:

                    # get AUC
                    try:
                        auc = roc_auc_score(batch_label.astype(np.int32), logits)
                    except:
                        auc = 0.00
                    
                    msg = build_msg(stage="Trn", epoch=epoch, iter=data_loader.batch_index,
                                    gstep=sess.run(model.global_step),
                                    loss=loss, logl=log_loss, regl=reg_loss, auc=auc)
                    
                    # write to file
                    print(msg, file=performance_writer)

                    # print performance every 100 batches
                    if sess.run(model.global_step) % 100 == 0:
                        print(msg)

                # add tensorboard summary [removed]

            test_msg, multi_order_feat, multi_order_attn = run_evaluation(
                sess=sess, data_loader=data_loader,
                epoch=epoch, model=model, validation=False)

            print(test_msg)
            print(test_msg, file=performance_writer)

    print("Training finished!")


def run_evaluation(sess, data_loader, model,
                   epoch=None,
                   validation=True):
    """
    Run validation or testing
    :return:
    """
    if validation:
        batch_generator = data_loader.generate_val_ivl()
    else:
        batch_generator = data_loader.generate_test_ivl()

    test_predictions = []
    test_logits = []
    test_labels = []
    sum_loss, sum_logloss = 0, 0

    multi_order_feat_list, multi_order_attn_list = [], []
    while True:
        try:
            ind, label = next(batch_generator)
            label = label.squeeze()
        except StopIteration:
            break
        batch_pred, batch_logits, batch_loss, batch_logloss, \
        batch_multi_order_feat, batch_multi_order_attn = sess.run(
            fetches=[
                model.pred,
                model.logits,
                model.loss,
                model.log_loss,
                model.stacked_multi_order_feat,
                model.multi_order_attn],

            feed_dict={
                model.X_ind: ind,
                model.label: label,
                model.is_training: False}
        )
        batch_logits = batch_logits[:, 1]

        test_predictions += batch_pred.tolist()
        test_logits += batch_logits.tolist()
        test_labels += label.astype(np.int32).tolist()
        sum_loss += np.sum(batch_loss)
        sum_logloss += np.sum(batch_logloss)
        multi_order_feat_list.append(batch_multi_order_feat)
        multi_order_attn_list.append(batch_multi_order_attn)

    mean_loss = sum_loss / len(test_predictions)
    auc = roc_auc_score(test_labels, test_logits)

    msg = build_msg(
        stage="Vld" if validation else "Tst",
        epoch=epoch if epoch is not None else 999,
        meanl=mean_loss, gstep=sess.run(model.global_step),
        loss=sum_loss, logl=sum_logloss, auc=auc)

    multi_order_attn = np.concatenate(multi_order_attn_list[:-1], axis=0)
    multi_order_feat = np.concatenate(multi_order_feat_list[:-1], axis=0)

    return msg, multi_order_attn, multi_order_feat


def main(argv):

    create_folder_tree(FLAGS.dataset)

    print("loading dataset ...")
    dl = DataLoader(dataset=FLAGS.dataset, batch_size=FLAGS.batch_size)

    model = InterprecsysBase(
        embedding_dim=FLAGS.embedding_size,
        field_size=dl.field_size,
        feature_size=dl.feature_size,
        learning_rate=FLAGS.learning_rate,
        batch_size=FLAGS.batch_size,
        num_block=FLAGS.num_block,
        num_head=FLAGS.num_head,
        attention_size=FLAGS.attention_size,
        attention_size_last=FLAGS.attention_size_last,
        temperature=FLAGS.temperature,
        dropout_rate=FLAGS.dropout_rate,
        regularization_weight=FLAGS.regularization_weight,
        highest_order=FLAGS.highest_order,
        pred_mlp_layers=FLAGS.pred_layers,
        random_seed=Constant.RANDOM_SEED
    )

    run_model(data_loader=dl, model=model, epochs=FLAGS.epoch)


if __name__ == '__main__':
    tf.app.run()
