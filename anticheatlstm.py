# Sloppy Keras code to train a very basic MLP
# Please have mercy on me.

from __future__ import print_function
import numpy as np
np.set_printoptions(suppress=True)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM, SimpleRNN, GRU
from keras.utils import np_utils
from keras.optimizers import SGD
import os, math, time

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name)
        print('Elapsed: %s' % (time.time() - self.tstart))

DATA = "D:\\Downloads\\HALData-master"

def categorical_probas_to_classes(p):
    return np.argmax(p, axis=1)

def probas_to_classes(y_pred):
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        return categorical_probas_to_classes(y_pred)
    return np.array([1 if p > 0.5 else 0 for p in y_pred])

def shuffle_in_unison_inplace(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

## This returns two vectors:
## examples, which looks like [[feature, feature, feature, feature],
#							   [features, feature, feature, feature]]
#  labels, which looks like [[0, 1], [1, 0]] ...
def parse(d, keyword, op, Y_fn=False):
	examples = []
	labels = []
	for f in os.listdir(d):
		if keyword in f:
			lines = [line.rstrip('\n') for line in open(os.path.join(d, f))]
			ts = 0
			init_ts = -1

			lastAttackOS = -1
			attacks = []

			for l in lines:
				if not l.startswith("#"): #and l.startswith("attackInfo"):
					s = l.split(',')
					if l.startswith("attackInfo"):
						if lastAttackOS < 0:
							lastAttackOS = long(s[11])
						else:
							v = [int(long(s[11]) - lastAttackOS), float(s[7]), float(s[8])]
							attacks.append(v)
							lastAttackOS = long(s[11])
					if len(attacks) == 20:
						examples.append(attacks)
						if Y_fn:
							labels.append(f)
						else:
							labels.append(op[:])
						attacks = []

	return np.asarray(examples), np.asarray(labels)


X_van, Y_van = parse(DATA + "\\Vanilla", ".csv", [1, 0])
X_hac, Y_hac = parse(DATA + "\\Hacks", ".csv", [0, 1])

print("Shuffling")
shuffle_in_unison_inplace(X_van, Y_van)
shuffle_in_unison_inplace(X_hac, Y_hac)

# Each class must have equal support
X_hac = X_hac[:len(X_van)]
Y_hac = Y_hac[:len(X_van)]

# print(X_van[:3])
# print("Hack:")
# print(X_hac[:3])
# exit()

print("Support for each class: ", len(X_van))


van_l = math.floor(len(X_van) * 1)
hac_l = math.floor(len(X_hac) * 1)

X_train = np.concatenate((X_van[:van_l], X_hac[:hac_l]), axis=0)
Y_train = np.concatenate((Y_van[:van_l], Y_hac[:hac_l]), axis=0)

X_test = np.concatenate((X_van[van_l:], X_hac[hac_l:]), axis=0)
Y_test = np.concatenate((Y_van[van_l:], Y_hac[hac_l:]), axis=0)

model = Sequential()
model.add(LSTM(32, input_dim=3, input_length=20))
model.add(Dense(2))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.load_weights('prelim_lstm.h5')

#model.fit(X_train, Y_train, nb_epoch=100, batch_size=16)

#model.save_weights('prelim_lstm.h5')

score = model.evaluate(X_test, Y_test, batch_size=16)

for i in range(len(score)):
	print(model.metrics_names[i], score[i])


#X_test, Y_test = parse("C:\\Users\\shrey\\Downloads\\Telegram Desktop", ".csv", [0, 1], Y_fn=True)
X_test, Y_test = parse("D:\\Downloads\\HALData-master\\UnidentifiedHacks", ".csv", [0, 1], Y_fn=True)

with Timer("perdiction"): # the spelling mistake is intentional
	pred_probas = model.predict_proba(X_test, batch_size=16)
pred = probas_to_classes(pred_probas)
actual = probas_to_classes(Y_test)

for p, a, c in zip(pred, Y_test, pred_probas):
	print(p, a, c)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])
