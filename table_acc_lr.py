# Reference: https://github.com/yawen-d/Logistic-Regression-on-MNIST-with-NumPy-from-Scratch
# slightly modified in terms of the loss function, regularization, etc.

import numpy as np
import h5py
import pandas as pd

def load_mnist(filename):
    # load MNIST data
    MNIST_data = h5py.File(filename, 'r')
    x_train = np.float32(MNIST_data['x_train'][:])
    y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
    x_test = np.float32(MNIST_data['x_test'][:])
    y_test = np.int32(np.array(MNIST_data['y_test'][:,0]))
    MNIST_data.close()
    return x_train,y_train,x_test,y_test

def initialize(num_inputs,num_classes):
    return np.random.rand(10, 785) / np.sqrt(10 * 785)

def softmax(z):
    # implement the softmax functions

    exp_list = np.exp(z)
    result = 1/sum(exp_list) * exp_list
    result = result.reshape((len(z),1))
    assert (result.shape == (len(z),1))
    return result

def neg_log_loss(pred, label):
    # implement the negative log loss
    loss = -np.log(pred[int(label)])
    return loss

def clip(arr, norm):
    original_norm = np.linalg.norm(arr)
    if original_norm <= norm:
        return arr
    else:
        return arr / original_norm * norm


def mini_batch_gradient(param, x_batch, y_batch, clip_norm):
    # implement the function to compute the mini batch gradient
    batch_size = x_batch.shape[0]
    grad_total = np.zeros((10, 785))

    for i in range(batch_size):
        x,y = x_batch[i],y_batch[i]
        x = x.reshape((785,1))
        E = np.zeros((10,1))
        E[y][0] = 1 
        pred = softmax(np.matmul(param, x))

        grad = - np.matmul(E - pred, x.reshape((1,785)))
        grad = clip(grad, clip_norm)
        
        grad_total += grad

    dgrad = grad_total / batch_size    

    return dgrad

def evaluate(param, x_data, y_data):
    # implement the evaluation function
    w = param.transpose()
    dist = np.array([np.squeeze(softmax(np.matmul(x_data[i], w))) for i in range(len(y_data))])

    result = np.argmax(dist,axis=1)
    accuracy = sum(result == y_data)/float(len(y_data))


    return accuracy

def train(param, hyp, x_train, y_train, x_test, y_test):
    # implement the train function
    num_epoches = hyp['num_epoches']
    batch_size = hyp['batch_size']
    learning_rate = hyp['learning_rate']
    sigma = hyp['sigma']
    reg = hyp['reg']
    grad_clip = hyp['half_grad_sens']
    mode = hyp['mode']
    
    if mode == "sgd":
        num_batch = int(x_train.shape[0]/batch_size)
        
        for epoch in range(num_epoches):

        
            # for each batch of train data
            for batch in range(num_batch):
                index = np.random.choice(x_train.shape[0], batch_size, replace = False)
                x_batch = x_train[index]
                y_batch = y_train[index]
    
                # calculate the gradient
                dgrad = mini_batch_gradient(param, x_batch, y_batch, grad_clip)
                
                # update the parameters with the learning rate
                param = (1 - learning_rate * reg) * param - learning_rate * (dgrad + sigma * np.random.randn(10, 785))
                
        train_accu = evaluate(param, x_train,y_train)
        test_accu = evaluate(param, x_test,y_test)

        print("Run with mode " + str(mode) + " with epochs = " + str(num_epoches) + " and regularizer = " + str(reg))
        return train_accu, test_accu
    elif mode == "cgd":
        num_batch = int(x_train.shape[0]/batch_size)
        for epoch in range(num_epoches):

            # for each batch of train data
            for batch in range(num_batch):
                x_batch = x_train[batch_size*batch:batch_size*(batch+1)]
                y_batch = y_train[batch_size*batch:batch_size*(batch+1)]
    
                # calculate the gradient
                dgrad = mini_batch_gradient(param, x_batch, y_batch, grad_clip)
                
                # update the parameters with the learning rate
                param = (1 - learning_rate * reg) * param - learning_rate * (dgrad + sigma * np.random.randn(10, 785))

        train_accu = evaluate(param,x_train,y_train)
        test_accu = evaluate(param,x_test,y_test)

        print("Run with mode " + str(mode) + " with epochs = " + str(num_epoches) + " and regularizer = " + str(reg))
        return train_accu, test_accu

# parameters
learning_rate = 0.05
batch_size = 1500
sigma = 1/100
epochs_array = [50, 100, 200]
regularizer_array = [0.002, 0.004, 0.002, 0.004]
data_clip_norm = 8
half_grad_sens = 5
num_iter = 10

# data frame to save accuracy results
train_multicolumn = pd.MultiIndex.from_arrays([[""] + ["Train Accuracy"] * len(regularizer_array),
                                         [""] + ["$\\nsgd$"] * (len(regularizer_array) // 2) + ["$\\nmgd$"] * (len(regularizer_array) // 2),
                                         ["$E \setminus \lambda$"] + [str(lam) for lam in regularizer_array]])
test_multicolumn = pd.MultiIndex.from_arrays([[""] + ["Test Accuracy"] * len(regularizer_array),
                                         [""] + ["$\\nsgd$"] * (len(regularizer_array) // 2) + ["$\\nmgd$"] * (len(regularizer_array) // 2),
                                         ["$E \setminus \lambda$"] + [str(lam) for lam in regularizer_array]])    

df_train_accuracy = pd.DataFrame(index = range(len(epochs_array)),
                                 columns = train_multicolumn)
df_test_accuracy = pd.DataFrame(index = range(len(epochs_array)),
                                columns = test_multicolumn)

ncol = len(regularizer_array) + 1

np.random.seed(1)

# main experiment
# takes 6-7 hours (with the given parameters) on a device with i7-13700H, 16GB RAM
for row in range(len(epochs_array)):
    epochs = epochs_array[row]
    df_train_accuracy.iloc[row, 0] = epochs
    df_test_accuracy.iloc[row, 0] = epochs
    for column in range(len(regularizer_array)):
        if column < (len(regularizer_array) // 2):
            mode = "sgd"
        else:
            mode = "cgd"
        
        regularizer = regularizer_array[column]
        
        train_accuracy_array = []
        test_accuracy_array = []
        
        for idx in range(num_iter):
            hyperpara = {
                "num_epoches" : epochs,
                "batch_size" : batch_size,
                "learning_rate" : learning_rate,
                "sigma": sigma,
                "reg": regularizer,
                "half_grad_sens": half_grad_sens,
                "mode": mode
            }

            x_train,y_train,x_test,y_test = load_mnist('MNISTdata.hdf5')
    
            for i in range(x_train.shape[0]):
                x_train[i] = clip(x_train[i], data_clip_norm)
            

            x_train_aug = np.hstack((x_train, np.ones((60000, 1))))
            x_test_aug = np.hstack((x_test, np.ones((10000, 1))))
    
            # initialize the parameters
            num_inputs = x_train_aug.shape[1]
            num_classes = len(set(y_train))
            param = initialize(num_inputs,num_classes)
    
            # train the model
            train_accu, test_accu = train(param,hyperpara,x_train_aug,y_train,x_test_aug,y_test)

            train_accuracy_array.append(100 * train_accu)
            test_accuracy_array.append(100 * test_accu)

            print("Iteration " + str(idx) + " with mode " + str(hyperpara['mode']) + ", regularizer " + str(hyperpara['reg']) + ", epochs " + str(epochs) +  " completed.")
    
        train_accuracy_mean = np.mean(train_accuracy_array)
        train_accuracy_sd = np.std(train_accuracy_array)

        test_accuracy_mean = np.mean(test_accuracy_array)
        test_accuracy_sd = np.std(test_accuracy_array)
    
        df_train_accuracy.iloc[row, column+1] = "{:.2f} $\pm$ {:.2f}".format(train_accuracy_mean, train_accuracy_sd)
        df_test_accuracy.iloc[row, column+1] = "{:.2f} $\pm$ {:.2f}".format(test_accuracy_mean, test_accuracy_sd)

print("==========================ExperimentDone==========================")

s = df_train_accuracy.style
s.hide(axis = "index")
s.to_latex("lr_table_trainacc.tex",
           encoding = "utf-8",
           position = "H",
           position_float = "centering",
           hrules = True,
           column_format = ncol * "r",
           multicol_align = "c",
           caption = "Train accuracy (\\%) of $\\nsgd$ and $\\nmgd$ for regularized logistic regression.",
           label = "tab:lr-train")

s = df_test_accuracy.style
s.hide(axis = "index")
s.to_latex("lr_table_testacc.tex",
           encoding = "utf-8",
           position = "H",
           position_float = "centering",
           hrules = True,
           column_format = ncol * "r",
           multicol_align = "c",
           caption = "Test accuracy (\\%) of $\\nsgd$ and $\\nmgd$ for regularized logistic regression.",
           label = "tab:lr-test")