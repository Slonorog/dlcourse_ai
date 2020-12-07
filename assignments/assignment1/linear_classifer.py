import numpy as np


# predictions - Z
def softmax(predictions):
    '''
    Computes probabilities from scores
    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) - classifier output
    Returns:
      probs, np array of the same shape as predictions - probability for every class, 0..1
    '''

    single = (predictions.ndim == 1)

    if single:
        predictions = predictions.reshape(1, predictions.shape[0])

    maximums = np.amax(predictions, axis=1).reshape(predictions.shape[0], 1)
    predictions_ts = predictions - maximums

    predictions_exp = np.exp(predictions_ts)
    sums = np.sum(predictions_exp, axis=1).reshape(predictions_exp.shape[0], 1)
    result = predictions_exp / sums

    if single:
        result = result.reshape(result.size)

    return result   # S


# probs - S
# target_index - y
def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss
    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) - probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) - index of the true class for given sample(s)
    Returns:
      loss: single value
    '''

    single = (probs.ndim == 1)

    if single:
        probs = probs.reshape(1, probs.shape[0])
        target_index = np.array([target_index])

    rows = np.arange(target_index.shape[0])
    cols = target_index

    return np.mean(-np.log(probs[rows, cols]))  # L


# predictions - Z
# target_index - y
def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions, including the gradient
    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) - classifier output
      target_index: np array of int, shape is (1) or (batch_size) - index of the true class for given sample(s)
    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''

    single = (predictions.ndim == 1)

    #print(single)
    if single:
        predictions = predictions.reshape(1, predictions.shape[0])
        target_index = np.array([target_index])

    probs = softmax(predictions)                    # S
    loss = cross_entropy_loss(probs, target_index)  # L

    indicator = np.zeros(probs.shape)
    indicator[np.arange(probs.shape[0]), target_index] = 1      # 1(y)
    dprediction = (probs - indicator) / predictions.shape[0]    # dL/dZ = (S - 1(y)) / N

    if single:
        dprediction = dprediction.reshape(dprediction.size)

    return loss, dprediction    # L, dL/dZ



def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    raise Exception("Not implemented!")

    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)

    # TODO implement prediction and gradient over W
    # Your final implementation shouldn't have any loops
    raise Exception("Not implemented!")
    
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            raise Exception("Not implemented!")

            # end
            print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)

        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops
        raise Exception("Not implemented!")

        return y_pred



                
                                                          

            

                
