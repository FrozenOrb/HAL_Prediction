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
			moves = []
			attacks = []
			lastattackos = -1
			lastmoveos = -1
			lines = [line.rstrip('\n') for line in open(os.path.join(d, f))]

			ll = 10

			for l in lines:
				if not l.startswith("#"): #and l.startswith("attackInfo"):
					s = l.split(',')
					if len(s) < 12:
						continue
					ts = long(s[11])
					#print(len(moves), len(attacks))
					if len(moves) > ll*6-1 and len(attacks) > ll*3-1:
						examples.append(moves[:ll*6] + attacks[:ll*3])
						if Y_fn:
							labels.append(f)
						else:
							labels.append(op[:])
						moves = moves[ll:]
						attacks = attacks[ll:]
					if l.startswith("attackInfo"):
						if lastattackos < 0:
							lastattackos = long(s[11])
						else:
							attacks.append(long(s[11]) - lastattackos)
							attacks.append(float(s[8]))
							attacks.append(float(s[7]))
							lastattackos = long(s[11])
					if l.startswith("move"):
						if lastmoveos < 0:
							lastmoveos = long(s[11])
						else:
							moves.append(long(s[11]) - lastmoveos)
							for i in [1,2,3,4,5]:
								moves.append(float(s[i]))
							lastmoveos = long(s[11])

	return np.asarray(examples), np.asarray(labels)


X_van, Y_van = parse(DATA + "\\Vanilla", ".csv", [1, 0])
X_hac, Y_hac = parse(DATA + "\\Hacks", ".csv", [0, 1])

# Each class must have equal support
X_hac = X_hac[:len(X_van)]
Y_hac = Y_hac[:len(X_van)]

# for i in range(10):
# 	print(len(X_van[i]))
# print("Hack:")
# for i in range(10):
# 	print(len(X_hac[i]))
# exit()

print("Support for van: ", len(X_van))
print("Support for hac: ", len(X_hac))

van_l = math.floor(len(X_van) * 0.8)
hac_l = math.floor(len(X_hac) * 0.8)

X_train = np.concatenate((X_van[:van_l], X_hac[:hac_l]), axis=0)
Y_train = np.concatenate((Y_van[:van_l], Y_hac[:hac_l]), axis=0)

X_test = np.concatenate((X_van[van_l:], X_hac[hac_l:]), axis=0)
Y_test = np.concatenate((Y_van[van_l:], Y_hac[hac_l:]), axis=0)

model = Sequential()
model.add(Dense(50, input_dim=90, init='uniform'))
model.add(Activation('tanh'))
model.add(Dense(20, init='uniform'))
model.add(Activation('tanh'))
model.add(Dense(2, init='uniform'))
model.add(Activation('softmax'))

sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.load_weights('prelim_massfeats_as.2.h5')

# model.fit(X_train, Y_train,
#          nb_epoch=200, batch_size=32)

# score = model.evaluate(X_test, Y_test, batch_size=32)

# for i in range(len(model.metrics_names)):
# 	print(model.metrics_names[i], score[i])

# model.save_weights('prelim_massfeats_as.2.h5')

#X_test, Y_test = parse("C:\\Users\\shrey\\Downloads\\Telegram Desktop", ".csv", [0, 1], Y_fn=True)
X_test, Y_test = parse("D:\\Downloads\\HALData-master\\Hacks", ".csv", [0, 1], Y_fn=True)

with Timer("perdiction"):
	pred_probas = model.predict_proba(X_test, batch_size=32)
pred = probas_to_classes(pred_probas)
actual = probas_to_classes(Y_test)

lst_c = {}

for p, a, c in zip(pred, Y_test, pred_probas):
	if c[1] > 0.8:
		print(p, a, c)
		#lst_c[a.split('-')[2]] = '1'

# for k in lst_c:
# 	print(k)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])

