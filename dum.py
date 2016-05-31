from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np
from keras.utils import np_utils

# Create a random matrix with 1000 rows (data points) and 15 columns (features)
train_rows = 1000
X_train = np.random.rand(train_rows, 15)

# Create a vector of 1000 random binary labels (one for wach row of X_train).
# It's a two class problem simulation, so a row can be class 0 or 1
labels = np.random.randint(2, size=train_rows)

# Now, the fit functions expects this labels to be encoded as one-hot vectors.
# In this case, this means we want a labels matrix with 50 rows, each row being
# [1, 0] (class 0) or [0, 1] (class 1).
# We'll use a util function to convert our labels vector to this format
y_train = np_utils.to_categorical(labels)

# Let's create some bogus test data also
test_rows = 500
X_test = np.random.rand(test_rows, 15)
labels = np.random.randint(2, size=test_rows)
y_test = np_utils.to_categorical(labels)


model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, input_dim=X_train.shape[1], init='uniform')) # X_train.shape[1] == 15 here
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(64, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(y_train.shape[1], init='uniform')) # y_train.shape[1] == 2 here
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

model.fit(X_train, y_train, nb_epoch=10, batch_size=100)
score = model.evaluate(X_test, y_test, batch_size=100)

# We'll achieve a score of approximately 0.25. That's because in our random data
# we have 50% = 0.5 of chance of getting the right answer, so the model will learn
# to predict probabilities near 0.5. But as we're using a mean square error,
# the score will be roughly 0.5^2 = 0.25
print "Score: %f" % score

# To see some predictions from the test set:
print 'Some predictions from the test set:'
print model.predict(X_test[0:10])