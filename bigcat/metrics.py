import keras.backend as K

def true_positives(y_true, y_pred):
    """True positive number metric (TP) for binary classification.
    Only computes a batch-wise average of TP.

    Args:
        y_true: The ground truth values.
            (y_true should be filled with 0 and 1 only) 
        y_pred: The predicted values.
    
    Returns:
        True positive number
    """
    y_true_positive = y_true
    y_pred_positive = K.round(K.clip(y_pred, 0, 1))
    true_positives = K.sum(y_true_positive * y_pred_positive)
    return true_positives

def false_positives(y_true, y_pred):
    """False positive number metric (FP) for binary classification.
    Only computes a batch-wise average of FP.
    
    Args:
        y_true: The ground truth values.
            (y_true should be filled with 0 and 1 only) 
        y_pred: The predicted values. 
    
    Returns:
        False positive number
    """
    y_true_negative = 1 - y_true
    y_pred_positive = K.round(K.clip(y_pred, 0, 1))
    false_positives = K.sum(y_true_negative * y_pred_positive)
    return false_positives

def true_negatives(y_true, y_pred):
    """True negative number metric (TN) for binary classification.
    Only computes a batch-wise average of TN.
    
    Args:
        y_true: The ground truth values.
            (y_true should be filled with 0 and 1 only) 
        y_pred: The predicted values. 
    
    Returns:
        True negative number
    """ 
    y_true_negative = 1 - y_true
    y_pred_negative = 1 - K.round(K.clip(y_pred, 0, 1))
    true_negatives = K.sum(y_pred_negative * y_true_negative)
    return true_negatives

def false_negatives(y_true, y_pred):
    """False negative number metric (FN) for binary classification.
    Only computes a batch-wise average of FN.
    
    Args:
        y_true: The ground truth values.
            (y_true should be filled with 0 and 1 only) 
        y_pred: The predicted values. 
    
    Returns:
        False negative number
    """
    y_true_positive = y_true
    y_pred_negative = 1 - K.round(K.clip(y_pred, 0, 1))
    true_negatives = K.sum(y_true_positive * y_pred_negative)
    return false_negatives

def precision(y_true, y_pred):
    """Precision metric for binary classification.
    Only computes a batch-wise average of precision.
    Code cloned from Keras
    
    Args:
        y_true: The ground truth values.
            (y_true should be filled with 0 and 1 only) 
        y_pred: The predicted values. 
    
    Returns:
        Precision
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) # equivalent to true_positives(y_true, y_pred) above
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon()) # add K.epsilon() to make this function stable computationally
    return precision

def recall(y_true, y_pred):
    """Recall metric for binary classification.
    Only computes a batch-wise average of recall.
    Code cloned from Keras
    
    Args:
        y_true: The ground truth values.
            (y_true should be filled with 0 and 1 only) 
        y_pred: The predicted values. 
    
    Returns:
        Recall
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # equivalent to true_positives(y_true, y_pred) above
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon()) # add K.epsilon() to make this function stable computationally
    return recall