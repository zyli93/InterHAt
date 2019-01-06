import tensorflow as tf

from interprecsys.old.data_reader import FeatureDictionary, DataParser
from .data_loader import DataLoader
from .const import Constant

from .model import Interprecsys, InterprecsysBase

flags = tf.app.flags

# TODO: Drop out!
# TODO: number of neuron's each layer
# TODO: what is batch norm? Do Batch Norm Somewhere. `Batch Norm Decay`

# Run time
flags.DEFINE_integer('epoch', 30, 'Number of Epochs.')
flags.DEFINE_integer('batch_size', 64, 'Number of training instance per batch.')
flags.DEFINE_string('dataset', 'example', 'Name of the dataset.')

# Optimization
# flags.DEFINE_string('optimizer', 'adam', 'Optimizer: adam/')  # TODO: more optimizer
# flags.DEFINE_string('activation', 'relu', 'Activation Layer: relu/')  # TODO: more activation
flags.DEFINE_float('learning_rate', 0.001, 'Learning Rate')
flags.DEFINE_float('l2_reg', 0.01, 'Weight of L2 Regularizations')

# Parameter Space
flags.DEFINE_integer('embedding_size', 256, 'Hidden Embedding Size')

# Hyper-param
flags.DEFINE_string('trial_id', '001', 'The ID of the current run.')
flags.DEFINE_float('entity_graph_threshold', 0.5, 'The threshold used when building subgraphs.')
flags.DEFINE_integer('neg_pos_ratio', 3, 'The ratio of negative samples v.s. positive.')
flags.DEFINE_float('dropout_rate', 0.1, 'The dropout rate of Transformer model.')
flags.DEFINE_float('regularization_weight', 0.01, 'The weight of L2-regularization.')

# Structure & Configure
flags.DEFINE_integer('random_seed', 2018, 'Random Seed.')
flags.DEFINE_integer('num_block', 2, 'Number of blocks of Multi-head Attention.')
flags.DEFINE_integer('num_head', 8, 'Number of heads of Multi-head Attention.')
flags.DEFINE_boolean('scale_embedding', True, 'Boolean. Whether scale the embeddings.')

# Options
flags.DEFINE_boolean('use_graph', True, 'Whether use graph information.')
flags.DEFINE.string('nct_neg_sample_method', 'uniform', 'Non click-through negative sampling method')

FLAGS = flags.FLAGS


def run_model(data_loader,
              model,
              epochs,
              is_training=True):
    """
    Run model (fit/predict)
    
    :param data_loader:
    :param model:
    :param epochs:
    :param is_training: True - Training; False - Evaluation.
    :return: 
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    """
    available outcomes:
        - predict
        - accuracy
        - loss
    """
    for epoch in range(epochs):
        data_loader.has_next = True
        while data_loader.has_next:

            batch_ind, batch_val = data_loader.generate_train_batch()
            feed_dict = {
                model.X_ind: batch_ind,
                model.X_val: batch_val
            }

            op, merged, loss = sess.run(
                fetches=[model.train_op, model.merged,
                         model.mean_loss],
                feed_dict=feed_dict
            )


def main():

    dl = DataLoader(dataset=FLAGS.dataset,
                    use_graph=FLAGS.use_graph,
                    entity_graph_threshold=FLAGS.entity_graph_threshold,
                    batch_size=FLAGS.batch_size)

    if FLAGS.use_graph:
        raise NotImplementedError("Graph version not implemented!")
    else:
        model = InterprecsysBase(
            embedding_dim=FLAGS.embedding_size,
            learning_rate=FLAGS.learning_rate,
            field_size=dl.field_size,
            feature_size=dl.feature_size,
            batch_size=FLAGS.batch_size,
            num_block=FLAGS.num_block,
            num_head=FLAGS.num_head,
            dropout_rate=FLAGS.dropout_rate,
            regularization_weight=FLAGS.regularization_weight,
            random_seed=Constant.RANDOM_SEED,
            scale_embedding=FLAGS.scale_embedding,
            is_training=True
        )

    # ===== Model training =====
    run_model(
        data_loader=dl,
        model=model,
        epochs=FLAGS.epoch,
        is_training=True
    )

    # ===== Model evaluation ======


if __name__ == '__main__':
    tf.app.run()

