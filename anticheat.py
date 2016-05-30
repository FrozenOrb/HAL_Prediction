# Sloppy Keras code to train a very basic MLP
# Please have mercy on me.

from __future__ import print_function
import numpy as np
np.set_printoptions(suppress=True)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
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



			for l in lines:
				if not l.startswith("#"): #and l.startswith("attackInfo"):
					s = l.split(',')
					if len(s) < 12:
						continue
					ts = long(s[11])
					if init_ts < 0:
						init_ts = ts
						num_attacks = 0
						max_dist = -1.0
						max_angle = -1.0
						vel_total = []
						vel_count = 0
					if l.startswith("attackInfo"):
						num_attacks += 1
						max_dist = max(float(s[8]), max_dist)
						max_angle = max(abs(float(s[7])), max_angle)
					if l.startswith("move"):
						if s[12].isdigit():
							if int(s[12]) != 0:
								vel_count += 1
								vel_total.append(abs(float(s[1]) / (float(s[12])/1000)))
								#print(abs(float(s[1]) / (float(s[12])/1000)), f, ts)
					if ts - init_ts > 5*1000 - 1:
						init_ts = -1
						if num_attacks != 0:
							examples.append([num_attacks / 10.0, max_angle, max_dist, np.median(vel_total)])
							if Y_fn:
								labels.append(f)
							else:
								labels.append(op[:])
	return np.asarray(examples), np.asarray(labels)


X_van, Y_van = parse(DATA + "\\Vanilla", ".csv", [1, 0])
X_hac, Y_hac = parse(DATA + "\\Hacks", ".csv", [0, 1])

# Each class must have equal support
X_hac = X_hac[:len(X_van)]
Y_hac = Y_hac[:len(X_van)]

# print(X_van)
# print("Hack:")
# print(X_hac)
# exit()

print("Support for each class: ", len(X_van))


van_l = math.floor(len(X_van) * 0.8)
hac_l = math.floor(len(X_hac) * 0.8)

X_train = np.concatenate((X_van[:van_l], X_hac[:hac_l]), axis=0)
Y_train = np.concatenate((Y_van[:van_l], Y_hac[:hac_l]), axis=0)

X_test = np.concatenate((X_van[van_l:], X_hac[hac_l:]), axis=0)
Y_test = np.concatenate((Y_van[van_l:], Y_hac[hac_l:]), axis=0)

model = Sequential()
model.add(Dense(20, input_dim=4, init='uniform'))
model.add(Activation('tanh'))
model.add(Dense(2, init='uniform'))
model.add(Activation('softmax'))

sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.load_weights('prelim_4feats_as.h5')

#model.fit(X_train, Y_train,
#          nb_epoch=500, batch_size=16)

score = model.evaluate(X_test, Y_test, batch_size=16)

for i in range(len(model.metrics_names)):
	print(model.metrics_names[i], score[i])

#model.save_weights('prelim_4feats_as.h5')

X_test, Y_test = parse("D:\\Downloads\\HALData-master\\UnidentifiedHacks", ".csv", [0, 1], Y_fn=True)

with Timer("perdiction"):
	pred_probas = model.predict_proba(X_test, batch_size=16)
pred = probas_to_classes(pred_probas)
actual = probas_to_classes(Y_test)

for p, a, c in zip(pred, Y_test, pred_probas):
	print(p, a, c)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])

