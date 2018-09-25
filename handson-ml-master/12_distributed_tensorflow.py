
# coding: utf-8

# **Chapter 12 – Distributed TensorFlow**

# _This notebook contains all the sample code and solutions to the exercises in chapter 12._

# # Setup

# First, let's make sure this notebook works well in both python 2 and 3, import a few common modules, ensure MatplotLib plots figures inline and prepare a function to save the figures:

# In[1]:


# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# To plot pretty figures
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "distributed"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


# # Local server

# In[2]:


import tensorflow as tf


# In[3]:


c = tf.constant("Hello distributed TensorFlow!")
server = tf.train.Server.create_local_server()


# In[4]:


with tf.Session(server.target) as sess:
    print(sess.run(c))


# # Cluster

# In[5]:


cluster_spec = tf.train.ClusterSpec({
    "ps": [
        "127.0.0.1:2221",  # /job:ps/task:0
        "127.0.0.1:2222",  # /job:ps/task:1
    ],
    "worker": [
        "127.0.0.1:2223",  # /job:worker/task:0
        "127.0.0.1:2224",  # /job:worker/task:1
        "127.0.0.1:2225",  # /job:worker/task:2
    ]})


# In[6]:


task_ps0 = tf.train.Server(cluster_spec, job_name="ps", task_index=0)
task_ps1 = tf.train.Server(cluster_spec, job_name="ps", task_index=1)
task_worker0 = tf.train.Server(cluster_spec, job_name="worker", task_index=0)
task_worker1 = tf.train.Server(cluster_spec, job_name="worker", task_index=1)
task_worker2 = tf.train.Server(cluster_spec, job_name="worker", task_index=2)


# # Pinning operations across devices and servers

# In[7]:


reset_graph()

with tf.device("/job:ps"):
    a = tf.Variable(1.0, name="a")

with tf.device("/job:worker"):
    b = a + 2

with tf.device("/job:worker/task:1"):
    c = a + b


# In[8]:


with tf.Session("grpc://127.0.0.1:2221") as sess:
    sess.run(a.initializer)
    print(c.eval())


# In[9]:


reset_graph()

with tf.device(tf.train.replica_device_setter(
        ps_tasks=2,
        ps_device="/job:ps",
        worker_device="/job:worker")):
    v1 = tf.Variable(1.0, name="v1")  # pinned to /job:ps/task:0 (defaults to /cpu:0)
    v2 = tf.Variable(2.0, name="v2")  # pinned to /job:ps/task:1 (defaults to /cpu:0)
    v3 = tf.Variable(3.0, name="v3")  # pinned to /job:ps/task:0 (defaults to /cpu:0)
    s = v1 + v2            # pinned to /job:worker (defaults to task:0/cpu:0)
    with tf.device("/task:1"):
        p1 = 2 * s         # pinned to /job:worker/task:1 (defaults to /cpu:0)
        with tf.device("/cpu:0"):
            p2 = 3 * s     # pinned to /job:worker/task:1/cpu:0

config = tf.ConfigProto()
config.log_device_placement = True

with tf.Session("grpc://127.0.0.1:2221", config=config) as sess:
    v1.initializer.run()


# # Readers – the old way

# In[10]:


reset_graph()


# In[11]:


default1 = tf.constant([5.])
default2 = tf.constant([6])
default3 = tf.constant([7])
dec = tf.decode_csv(tf.constant("1.,,44"),
                    record_defaults=[default1, default2, default3])
with tf.Session() as sess:
    print(sess.run(dec))


# In[12]:


reset_graph()

test_csv = open("my_test.csv", "w")
test_csv.write("x1, x2 , target\n")
test_csv.write("1.,, 0\n")
test_csv.write("4., 5. , 1\n")
test_csv.write("7., 8. , 0\n")
test_csv.close()

filename_queue = tf.FIFOQueue(capacity=10, dtypes=[tf.string], shapes=[()])
filename = tf.placeholder(tf.string)
enqueue_filename = filename_queue.enqueue([filename])
close_filename_queue = filename_queue.close()

reader = tf.TextLineReader(skip_header_lines=1)
key, value = reader.read(filename_queue)

x1, x2, target = tf.decode_csv(value, record_defaults=[[-1.], [-1.], [-1]])
features = tf.stack([x1, x2])

instance_queue = tf.RandomShuffleQueue(
    capacity=10, min_after_dequeue=2,
    dtypes=[tf.float32, tf.int32], shapes=[[2],[]],
    name="instance_q", shared_name="shared_instance_q")
enqueue_instance = instance_queue.enqueue([features, target])
close_instance_queue = instance_queue.close()

minibatch_instances, minibatch_targets = instance_queue.dequeue_up_to(2)

with tf.Session() as sess:
    sess.run(enqueue_filename, feed_dict={filename: "my_test.csv"})
    sess.run(close_filename_queue)
    try:
        while True:
            sess.run(enqueue_instance)
    except tf.errors.OutOfRangeError as ex:
        print("No more files to read")
    sess.run(close_instance_queue)
    try:
        while True:
            print(sess.run([minibatch_instances, minibatch_targets]))
    except tf.errors.OutOfRangeError as ex:
        print("No more training instances")


# In[13]:


#coord = tf.train.Coordinator()
#threads = tf.train.start_queue_runners(coord=coord)
#filename_queue = tf.train.string_input_producer(["test.csv"])
#coord.request_stop()
#coord.join(threads)


# # Queue runners and coordinators

# In[14]:


reset_graph()

filename_queue = tf.FIFOQueue(capacity=10, dtypes=[tf.string], shapes=[()])
filename = tf.placeholder(tf.string)
enqueue_filename = filename_queue.enqueue([filename])
close_filename_queue = filename_queue.close()

reader = tf.TextLineReader(skip_header_lines=1)
key, value = reader.read(filename_queue)

x1, x2, target = tf.decode_csv(value, record_defaults=[[-1.], [-1.], [-1]])
features = tf.stack([x1, x2])

instance_queue = tf.RandomShuffleQueue(
    capacity=10, min_after_dequeue=2,
    dtypes=[tf.float32, tf.int32], shapes=[[2],[]],
    name="instance_q", shared_name="shared_instance_q")
enqueue_instance = instance_queue.enqueue([features, target])
close_instance_queue = instance_queue.close()

minibatch_instances, minibatch_targets = instance_queue.dequeue_up_to(2)

n_threads = 5
queue_runner = tf.train.QueueRunner(instance_queue, [enqueue_instance] * n_threads)
coord = tf.train.Coordinator()

with tf.Session() as sess:
    sess.run(enqueue_filename, feed_dict={filename: "my_test.csv"})
    sess.run(close_filename_queue)
    enqueue_threads = queue_runner.create_threads(sess, coord=coord, start=True)
    try:
        while True:
            print(sess.run([minibatch_instances, minibatch_targets]))
    except tf.errors.OutOfRangeError as ex:
        print("No more training instances")


# In[15]:


reset_graph()

def read_and_push_instance(filename_queue, instance_queue):
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)
    x1, x2, target = tf.decode_csv(value, record_defaults=[[-1.], [-1.], [-1]])
    features = tf.stack([x1, x2])
    enqueue_instance = instance_queue.enqueue([features, target])
    return enqueue_instance

filename_queue = tf.FIFOQueue(capacity=10, dtypes=[tf.string], shapes=[()])
filename = tf.placeholder(tf.string)
enqueue_filename = filename_queue.enqueue([filename])
close_filename_queue = filename_queue.close()

instance_queue = tf.RandomShuffleQueue(
    capacity=10, min_after_dequeue=2,
    dtypes=[tf.float32, tf.int32], shapes=[[2],[]],
    name="instance_q", shared_name="shared_instance_q")

minibatch_instances, minibatch_targets = instance_queue.dequeue_up_to(2)

read_and_enqueue_ops = [read_and_push_instance(filename_queue, instance_queue) for i in range(5)]
queue_runner = tf.train.QueueRunner(instance_queue, read_and_enqueue_ops)

with tf.Session() as sess:
    sess.run(enqueue_filename, feed_dict={filename: "my_test.csv"})
    sess.run(close_filename_queue)
    coord = tf.train.Coordinator()
    enqueue_threads = queue_runner.create_threads(sess, coord=coord, start=True)
    try:
        while True:
            print(sess.run([minibatch_instances, minibatch_targets]))
    except tf.errors.OutOfRangeError as ex:
        print("No more training instances")


# # Setting a timeout

# In[16]:


reset_graph()

q = tf.FIFOQueue(capacity=10, dtypes=[tf.float32], shapes=[()])
v = tf.placeholder(tf.float32)
enqueue = q.enqueue([v])
dequeue = q.dequeue()
output = dequeue + 1

config = tf.ConfigProto()
config.operation_timeout_in_ms = 1000

with tf.Session(config=config) as sess:
    sess.run(enqueue, feed_dict={v: 1.0})
    sess.run(enqueue, feed_dict={v: 2.0})
    sess.run(enqueue, feed_dict={v: 3.0})
    print(sess.run(output))
    print(sess.run(output, feed_dict={dequeue: 5}))
    print(sess.run(output))
    print(sess.run(output))
    try:
        print(sess.run(output))
    except tf.errors.DeadlineExceededError as ex:
        print("Timed out while dequeuing")


# # Data API

# The Data API, introduced in TensorFlow 1.4, makes reading data efficiently much easier.

# In[17]:


tf.reset_default_graph()


# Let's start with a simple dataset composed of three times the integers 0 to 9, in batches of 7:

# In[18]:


dataset = tf.data.Dataset.from_tensor_slices(np.arange(10))
dataset = dataset.repeat(3).batch(7)


# The first line creates a dataset containing the integers 0 through 9. The second line creates a new dataset based on the first one, repeating its elements three times and creating batches of 7 elements. As you can see, we start with a source dataset, then we chain calls to various methods to apply transformations to the data.

# Next, we create a one-shot-iterator to go through this dataset just once, and we call its `get_next()` method to get a tensor that represents the next element.

# In[19]:


iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()


# Let's repeatedly evaluate `next_element` to go through the dataset. When there are not more elements, we get an `OutOfRangeError`:

# In[20]:


with tf.Session() as sess:
    try:
        while True:
            print(next_element.eval())
    except tf.errors.OutOfRangeError:
        print("Done")


# Great! It worked fine.

# Note that, as always, a tensor is only evaluated once each time we run the graph (`sess.run()`): so even if we evaluate multiple tensors that all depend on `next_element`, it is only evaluated once. This is true as well if we ask for `next_element` to be evaluated twice in just one run:

# In[21]:


with tf.Session() as sess:
    try:
        while True:
            print(sess.run([next_element, next_element]))
    except tf.errors.OutOfRangeError:
        print("Done")


# The `interleave()` method is powerful but a bit tricky to grasp at first. The easiest way to understand it is to look at an example:

# In[22]:


tf.reset_default_graph()


# In[23]:


dataset = tf.data.Dataset.from_tensor_slices(np.arange(10))
dataset = dataset.repeat(3).batch(7)
dataset = dataset.interleave(
    lambda v: tf.data.Dataset.from_tensor_slices(v),
    cycle_length=3,
    block_length=2)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()


# In[24]:


with tf.Session() as sess:
    try:
        while True:
            print(next_element.eval(), end=",")
    except tf.errors.OutOfRangeError:
        print("Done")


# Because `cycle_length=3`, the new dataset starts by pulling 3 elements from the previous dataset: that's `[0,1,2,3,4,5,6]`, `[7,8,9,0,1,2,3]` and `[4,5,6,7,8,9,0]`. Then it calls the lambda function we gave it to create one dataset for each of the elements. Since we use `Dataset.from_tensor_slices()`, each dataset is going to return its elements one by one. Next, it pulls two items (since `block_length=2`) from each of these three datasets, and it iterates until all three datasets are out of items: 0,1 (from 1st), 7,8 (from 2nd), 4,5 (from 3rd), 2,3 (from 1st), 9,0 (from 2nd), and so on until 8,9 (from 3rd), 6 (from 1st), 3 (from 2nd), 0 (from 3rd). Next it tries to pull the next 3 elements from the original dataset, but there are just two left: `[1,2,3,4,5,6,7]` and `[8,9]`. Again, it creates datasets from these elements, and it pulls two items from each until both datasets are out of items: 1,2 (from 1st), 8,9 (from 2nd), 3,4 (from 1st), 5,6 (from 1st), 7 (from 1st). Notice that there's no interleaving at the end since the arrays do not have the same length.

# # Readers – the new way

# Instead of using a source dataset based on `from_tensor_slices()` or `from_tensor()`, we can use a reader dataset. It handles most of the complexity for us (e.g., threads):

# In[25]:


tf.reset_default_graph()


# In[26]:


filenames = ["my_test.csv"]


# In[27]:


dataset = tf.data.TextLineDataset(filenames)


# We still need to tell it how to decode each line:

# In[28]:


def decode_csv_line(line):
    x1, x2, y = tf.decode_csv(
        line, record_defaults=[[-1.], [-1.], [-1.]])
    X = tf.stack([x1, x2])
    return X, y


# Next, we can apply this decoding function to each element in the dataset using `map()`:

# In[29]:


dataset = dataset.skip(1).map(decode_csv_line)


# Finally, let's create a one-shot iterator:

# In[30]:


it = dataset.make_one_shot_iterator()
X, y = it.get_next()


# In[31]:


with tf.Session() as sess:
    try:
        while True:
            X_val, y_val = sess.run([X, y])
            print(X_val, y_val)
    except tf.errors.OutOfRangeError as ex:
        print("Done")


# # Exercise solutions

# **Coming soon**
