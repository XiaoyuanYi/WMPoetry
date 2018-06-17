from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.util import nest
from tensorflow.python.framework import ops

def sequence_loss_by_example(logits, targets, weights,
                             average_across_timesteps=True,
                             softmax_loss_function=None, name=None):
    """Weighted cross-entropy loss for a sequence of logits (per example).

    Args:
        logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
        targets: List of 1D batch-sized int32 Tensors of the same length as logits.
        weights: List of 1D batch-sized float-Tensors of the same length as logits.
        average_across_timesteps: If set, divide the returned cost by the total label weight.
        softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
          to be used instead of the standard softmax (the default if this is None).
        name: Optional name for this operation, default: "sequence_loss_by_example".

        Returns:
            1D batch-sized float Tensor: The log-perplexity for each sequence.

        Raises:
            ValueError: If len(logits) is different from len(targets) or len(weights).
    """
    if len(targets) != len(logits) or len(weights) != len(logits):
        raise ValueError("Lengths of logits, weights, and targets must be the same "
            "%d, %d, %d." % (len(logits), len(weights), len(targets)))

    with ops.name_scope("sequence_loss_by_example"):
        log_perp_list = []
        for logit, target, weight in zip(logits, targets, weights):
            if softmax_loss_function is None:
                target = array_ops.reshape(target, [-1])
                crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=target)
            else:
                crossent = softmax_loss_function(logit, target)
            log_perp_list.append(crossent * weight)

        log_perps = math_ops.add_n(log_perp_list)
        if average_across_timesteps:
            total_size = math_ops.add_n(weights)
            total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
            log_perps /= total_size

    return log_perps

def sequence_loss(logits, targets, weights,
                  average_across_timesteps=True, average_across_batch=True,
                  softmax_loss_function=None, name=None):

    with ops.name_scope("sequence_loss"):
        cost = math_ops.reduce_sum(sequence_loss_by_example(logits, targets, weights,
        average_across_timesteps=average_across_timesteps,
        softmax_loss_function=softmax_loss_function))

        if average_across_batch:
            batch_size = array_ops.shape(targets[0])[0]
            return cost / math_ops.cast(batch_size, cost.dtype)
        else:
            return cost

def flatten_query(query):
    """Put attention masks on hidden using hidden_features and query."""
    if nest.is_sequence(query):  # If the query is a tuple, flatten it.
        query_list = nest.flatten(query)
        for q in query_list:  # Check that ndims == 2 if specified.
            ndims = q.get_shape().ndims
            if ndims:
                assert ndims == 2
        return array_ops.concat(query_list, 1)
    else:
        return query
