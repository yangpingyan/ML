
# coding: utf-8

# # TensorFlow Reproducibility

# In[1]:


from __future__ import division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
from tensorflow import keras


# ## Checklist

# 1. Do not run TensorFlow on the GPU.
# 2. Beware of multithreading, and make TensorFlow single-threaded.
# 3. Set all the random seeds.
# 4. Eliminate any other source of variability.

# ## Do Not Run TensorFlow on the GPU

# Some operations (like `tf.reduce_sum()`) have favor performance over precision, and their outputs may vary slightly across runs. To get reproducible results, make sure TensorFlow runs on the CPU:

# In[2]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]=""


# ## Beware of Multithreading

# Because floats have limited precision, the order of execution matters:

# In[3]:


2. * 5. / 7.


# In[4]:


2. / 7. * 5.


# You should make sure TensorFlow runs your ops on a single thread:

# In[5]:


config = tf.ConfigProto(intra_op_parallelism_threads=1,
                        inter_op_parallelism_threads=1)

with tf.Session(config=config) as sess:
    #... this will run single threaded
    pass


# The thread pools for all sessions are created when you create the first session, so all sessions in the rest of this notebook will be single-threaded:

# In[6]:


with tf.Session() as sess:
    #... also single-threaded!
    pass


# ## Set all the random seeds!

# ### Python's built-in `hash()` function

# In[7]:


print(set("Try restarting the kernel and running this again"))
print(set("Try restarting the kernel and running this again"))


# Since Python 3.3, the result will be different every time, unless you start Python with the `PYTHONHASHSEED` environment variable set to `0`:

# ```shell
# PYTHONHASHSEED=0 python
# ```
# 
# ```pycon
# >>> print(set("Now the output is stable across runs"))
# {'n', 'b', 'h', 'o', 'i', 'a', 'r', 't', 'p', 'N', 's', 'c', ' ', 'l', 'e', 'w', 'u'}
# >>> exit()
# ```
# 
# ```shell
# PYTHONHASHSEED=0 python
# ```
# ```pycon
# >>> print(set("Now the output is stable across runs"))
# {'n', 'b', 'h', 'o', 'i', 'a', 'r', 't', 'p', 'N', 's', 'c', ' ', 'l', 'e', 'w', 'u'}
# ```

# Alternatively, you could set this environment variable system-wide, but that's probably not a good idea, because this automatic randomization was [introduced for security reasons](http://ocert.org/advisories/ocert-2011-003.html).

# Unfortunately, setting the environment variable from within Python (e.g., using `os.environ["PYTHONHASHSEED"]="0"`) will not work, because Python reads it upon startup. For Jupyter notebooks, you have to start the Jupyter server like this:
# 
# ```shell
# PYTHONHASHSEED=0 jupyter notebook
# ```

# In[8]:


if os.environ.get("PYTHONHASHSEED") != "0":
    raise Exception("You must set PYTHONHASHSEED=0 when starting the Jupyter server to get reproducible results.")


# ### Python Random Number Generators (RNGs)

# In[9]:


import random

random.seed(42)
print(random.random())
print(random.random())

print()

random.seed(42)
print(random.random())
print(random.random())


# ### NumPy RNGs

# In[10]:


import numpy as np

np.random.seed(42)
print(np.random.rand())
print(np.random.rand())

print()

np.random.seed(42)
print(np.random.rand())
print(np.random.rand())


# ### TensorFlow RNGs

# TensorFlow's behavior is more complex because of two things:
# * you create a graph, and then you execute it. The random seed must be set before you create the random operations.
# * there are two seeds: one at the graph level, and one at the individual random operation level.

# In[11]:


import tensorflow as tf

tf.set_random_seed(42)
rnd = tf.random_uniform(shape=[])

with tf.Session() as sess:
    print(rnd.eval())
    print(rnd.eval())

print()

with tf.Session() as sess:
    print(rnd.eval())
    print(rnd.eval())


# Every time you reset the graph, you need to set the seed again:

# In[12]:


tf.reset_default_graph()

tf.set_random_seed(42)
rnd = tf.random_uniform(shape=[])

with tf.Session() as sess:
    print(rnd.eval())
    print(rnd.eval())

print()

with tf.Session() as sess:
    print(rnd.eval())
    print(rnd.eval())


# If you create your own graph, it will ignore the default graph's seed:

# In[13]:


tf.reset_default_graph()
tf.set_random_seed(42)

graph = tf.Graph()
with graph.as_default():
    rnd = tf.random_uniform(shape=[])

with tf.Session(graph=graph):
    print(rnd.eval())
    print(rnd.eval())

print()

with tf.Session(graph=graph):
    print(rnd.eval())
    print(rnd.eval())


# You must set its own seed:

# In[14]:


graph = tf.Graph()
with graph.as_default():
    tf.set_random_seed(42)
    rnd = tf.random_uniform(shape=[])

with tf.Session(graph=graph):
    print(rnd.eval())
    print(rnd.eval())

print()

with tf.Session(graph=graph):
    print(rnd.eval())
    print(rnd.eval())


# If you set the seed after the random operation is created, the seed has no effet:

# In[15]:


tf.reset_default_graph()

rnd = tf.random_uniform(shape=[])

tf.set_random_seed(42) # BAD, NO EFFECT!
with tf.Session() as sess:
    print(rnd.eval())
    print(rnd.eval())

print()

tf.set_random_seed(42) # BAD, NO EFFECT!
with tf.Session() as sess:
    print(rnd.eval())
    print(rnd.eval())


# #### A note about operation seeds

# You can also set a seed for each individual random operation. When you do, it is combined with the graph seed into the final seed used by that op. The following table summarizes how this works:
# 
# | Graph seed | Op seed |  Resulting seed                |
# |------------|---------|--------------------------------|
# | None       | None    | Random                         |
# | graph_seed | None    | f(graph_seed, op_index)        |
# | None       | op_seed | f(default_graph_seed, op_seed) |
# | graph_seed | op_seed | f(graph_seed, op_seed)         |
# 
# * `f()` is a deterministic function.
# * `op_index = graph._last_id` when there is a graph seed, different random ops without op seeds will have different outputs. However, each of them will have the same sequence of outputs at every run.
# 
# In eager mode, there is a global seed instead of graph seed (since there is no graph in eager mode).

# In[16]:


tf.reset_default_graph()

rnd1 = tf.random_uniform(shape=[], seed=42)
rnd2 = tf.random_uniform(shape=[], seed=42)
rnd3 = tf.random_uniform(shape=[])

with tf.Session() as sess:
    print(rnd1.eval())
    print(rnd2.eval())
    print(rnd3.eval())
    print(rnd1.eval())
    print(rnd2.eval())
    print(rnd3.eval())

print()

with tf.Session() as sess:
    print(rnd1.eval())
    print(rnd2.eval())
    print(rnd3.eval())
    print(rnd1.eval())
    print(rnd2.eval())
    print(rnd3.eval())


# In the following example, you may think that all random ops will have the same random seed, but `rnd3` will actually have a different seed:

# In[17]:


tf.reset_default_graph()

tf.set_random_seed(42)

rnd1 = tf.random_uniform(shape=[], seed=42)
rnd2 = tf.random_uniform(shape=[], seed=42)
rnd3 = tf.random_uniform(shape=[])

with tf.Session() as sess:
    print(rnd1.eval())
    print(rnd2.eval())
    print(rnd3.eval())
    print(rnd1.eval())
    print(rnd2.eval())
    print(rnd3.eval())

print()

with tf.Session() as sess:
    print(rnd1.eval())
    print(rnd2.eval())
    print(rnd3.eval())
    print(rnd1.eval())
    print(rnd2.eval())
    print(rnd3.eval())


# #### Estimators API

# **Tip**: in a Jupyter notebook, you probably want to set the random seeds regularly so that you can come back and run the notebook from there (instead of from the beginning) and still get reproducible outputs.

# In[18]:


random.seed(42)
np.random.seed(42)
tf.set_random_seed(42)


# If you use the Estimators API, make sure to create a `RunConfig` and set its `tf_random_seed`, then pass it to the constructor of your estimator:

# In[19]:


my_config = tf.estimator.RunConfig(tf_random_seed=42)

feature_cols = [tf.feature_column.numeric_column("X", shape=[28 * 28])]
dnn_clf = tf.estimator.DNNClassifier(hidden_units=[300, 100], n_classes=10,
                                     feature_columns=feature_cols,
                                     config=my_config)


# Let's try it on MNIST:

# In[20]:


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)


# Unfortunately, the `numpy_input_fn` does not allow us to set the seed when `shuffle=True`, so we must shuffle the data ourself and set `shuffle=False`.

# In[21]:


indices = np.random.permutation(len(X_train))
X_train_shuffled = X_train[indices]
y_train_shuffled = y_train[indices]

input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"X": X_train_shuffled}, y=y_train_shuffled, num_epochs=10, batch_size=32, shuffle=False)
dnn_clf.train(input_fn=input_fn)


# The final loss should be exactly 0.04650943.

# Instead of using the `numpy_input_fn()` function (which cannot reproducibly shuffle the dataset at each epoch), you can create your own input function using the Data API and set its shuffling seed:

# In[22]:


def create_dataset(X, y=None, n_epochs=1, batch_size=32,
                   buffer_size=1000, seed=None):
    dataset = tf.data.Dataset.from_tensor_slices(({"X": X}, y))
    dataset = dataset.repeat(n_epochs)
    dataset = dataset.shuffle(buffer_size, seed=seed)
    return dataset.batch(batch_size)

input_fn=lambda: create_dataset(X_train, y_train, seed=42)


# In[23]:


random.seed(42)
np.random.seed(42)
tf.set_random_seed(42)

my_config = tf.estimator.RunConfig(tf_random_seed=42)

feature_cols = [tf.feature_column.numeric_column("X", shape=[28 * 28])]
dnn_clf = tf.estimator.DNNClassifier(hidden_units=[300, 100], n_classes=10,
                                     feature_columns=feature_cols,
                                     config=my_config)
dnn_clf.train(input_fn=input_fn)


# The final loss should be exactly 1.0556093.

# ```python
# indices = np.random.permutation(len(X_train))
# X_train_shuffled = X_train[indices]
# y_train_shuffled = y_train[indices]
# 
# input_fn = tf.estimator.inputs.numpy_input_fn(
#     x={"X": X_train_shuffled}, y=y_train_shuffled,
#     num_epochs=10, batch_size=32, shuffle=False)
# dnn_clf.train(input_fn=input_fn)
# ```

# #### Keras API

# If you use the Keras API, all you need to do is set the random seed any time you clear the session:

# In[24]:


keras.backend.clear_session()

random.seed(42)
np.random.seed(42)
tf.set_random_seed(42)

model = keras.models.Sequential([
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax"),
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd",
              metrics=["accuracy"])
model.fit(X_train, y_train, epochs=10)


# You should get exactly 97.16% accuracy on the training set at the end of training.

# ## Eliminate other sources of variability

# For example, `os.listdir()` returns file names in an order that depends on how the files were indexed by the file system:

# In[25]:


for i in range(10):
    with open("my_test_foo_{}".format(i), "w"):
        pass

[f for f in os.listdir() if f.startswith("my_test_foo_")]


# In[26]:


for i in range(10):
    with open("my_test_bar_{}".format(i), "w"):
        pass

[f for f in os.listdir() if f.startswith("my_test_bar_")]


# You should sort the file names before you use them:

# In[27]:


filenames = os.listdir()
filenames.sort()


# In[28]:


[f for f in filenames if f.startswith("my_test_foo_")]


# In[29]:


for f in os.listdir():
    if f.startswith("my_test_foo_") or f.startswith("my_test_bar_"):
        os.remove(f)


# I hope you enjoyed this notebook. If you do not get reproducible results, or if they are different than mine, then please [file an issue](https://github.com/ageron/handson-ml/issues) on github, specifying what version of Python, TensorFlow, and NumPy you are using, as well as your O.S. version. Thank you!

# If you want to learn more about Deep Learning and TensorFlow, check out my book [Hands-On Machine Learning with Scitkit-Learn and TensorFlow](http://homl.info/amazon), O'Reilly. You can also follow me on twitter [@aureliengeron](https://twitter.com/aureliengeron) or watch my videos on YouTube at [youtube.com/c/AurelienGeron](https://www.youtube.com/c/AurelienGeron).
