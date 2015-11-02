# Feel free to use numpy in your MLP if you like to.
import numpy as np
import mlp
import random

filename = 'movements_day1-3.dat'

movements = np.loadtxt(filename,delimiter='\t')
movements[:,:40] = movements[:,:40]-movements[:,:40].mean(axis=0)
imax = np.concatenate((movements.max(axis=0)*np.ones((1,41)),np.abs(movements.min(axis=0)*np.ones((1,41)))),axis=0).max(axis=0)
movements[:,:40] = movements[:,:40]/imax[:40]

# Split into training, validation, and test sets
target = np.zeros((np.shape(movements)[0],8));
for x in range(1,9):
    indices = np.where(movements[:,40]==x)
    target[indices,x-1] = 1


# Randomly order the data
order = range(np.shape(movements)[0])
np.random.shuffle(order)
movements = movements[order,:]
target = target[order,:]

train = movements[::2,0:40]
traint = target[::2]

valid = movements[1::4,0:40]
validt = target[1::4]

test = movements[3::4,0:40]
testt = target[3::4]

# Train the network
k_fold = False
fold_number = 4

if not k_fold:
    net = mlp.mlp(train,traint)
    net.earlystopping(train, traint, valid, validt)
    net.confusion(test,testt)
else:
    movements = movements[::,0:40]
    movements = movements.tolist()
    target = target.tolist()
    folds = []
    tfolds = []
    fold_size = len(movements) / fold_number
    for i in range(0, fold_number):
        fold = []
        tfold = []
        for j in range(0, fold_size):
            select = random.randint(0, len(movements) - 1)
            fold.append(movements[select])
            tfold.append(target[select])
            del movements[select]
            del target[select]

        folds.append(fold)
        tfolds.append(tfold)
    count = 0

    for i in range(0, fold_number):
        f_valid = folds[i]
        f_validt = tfolds[i]
        for j in range(0, fold_number):
            if i != j:
                f_test = folds[j]
                f_testt = tfolds[j]
                f_train = []
                f_traint = []
                for k in range(0, fold_number):
                    if k != i and k != j:
                        f_train += folds[k]
                        f_traint += tfolds[k]

                count += 1
                print "fold test ", count
                net = mlp.mlp(np.array(f_train),np.array(f_traint))
                net.earlystopping(np.array(f_train), np.array(f_traint), np.array(f_valid), np.array(f_validt))
                net.confusion(np.array(f_test),np.array(f_testt))