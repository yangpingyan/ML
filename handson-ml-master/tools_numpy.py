
# coding: utf-8

# **Tools - NumPy**
# 
# *NumPy is the fundamental library for scientific computing with Python. NumPy is centered around a powerful N-dimensional array object, and it also contains useful linear algebra, Fourier transform, and random number functions.*
# 
# # Creating arrays
# First let's make sure that this notebook works both in python 2 and 3:

# In[1]:


from __future__ import division, print_function, unicode_literals


# Now let's import `numpy`. Most people import it as `np`:

# In[2]:


import numpy as np


# ## `np.zeros`

# The `zeros` function creates an array containing any number of zeros:

# In[3]:


np.zeros(5)


# It's just as easy to create a 2D array (ie. a matrix) by providing a tuple with the desired number of rows and columns. For example, here's a 3x4 matrix:

# In[4]:


np.zeros((3,4))


# ## Some vocabulary
# 
# * In NumPy, each dimension is called an **axis**.
# * The number of axes is called the **rank**.
#     * For example, the above 3x4 matrix is an array of rank 2 (it is 2-dimensional).
#     * The first axis has length 3, the second has length 4.
# * An array's list of axis lengths is called the **shape** of the array.
#     * For example, the above matrix's shape is `(3, 4)`.
#     * The rank is equal to the shape's length.
# * The **size** of an array is the total number of elements, which is the product of all axis lengths (eg. 3*4=12)

# In[5]:


a = np.zeros((3,4))
a


# In[6]:


a.shape


# In[7]:


a.ndim  # equal to len(a.shape)


# In[8]:


a.size


# ## N-dimensional arrays
# You can also create an N-dimensional array of arbitrary rank. For example, here's a 3D array (rank=3), with shape `(2,3,4)`:

# In[9]:


np.zeros((2,3,4))


# ## Array type
# NumPy arrays have the type `ndarray`s:

# In[10]:


type(np.zeros((3,4)))


# ## `np.ones`
# Many other NumPy functions create `ndarrays`.
# 
# Here's a 3x4 matrix full of ones:

# In[11]:


np.ones((3,4))


# ## `np.full`
# Creates an array of the given shape initialized with the given value. Here's a 3x4 matrix full of `π`.

# In[12]:


np.full((3,4), np.pi)


# ## `np.empty`
# An uninitialized 2x3 array (its content is not predictable, as it is whatever is in memory at that point):

# In[13]:


np.empty((2,3))


# ## np.array
# Of course you can initialize an `ndarray` using a regular python array. Just call the `array` function:

# In[14]:


np.array([[1,2,3,4], [10, 20, 30, 40]])


# ## `np.arange`
# You can create an `ndarray` using NumPy's `range` function, which is similar to python's built-in `range` function:

# In[15]:


np.arange(1, 5)


# It also works with floats:

# In[16]:


np.arange(1.0, 5.0)


# Of course you can provide a step parameter:

# In[17]:


np.arange(1, 5, 0.5)


# However, when dealing with floats, the exact number of elements in the array is not always predictible. For example, consider this:

# In[18]:


print(np.arange(0, 5/3, 1/3)) # depending on floating point errors, the max value is 4/3 or 5/3.
print(np.arange(0, 5/3, 0.333333333))
print(np.arange(0, 5/3, 0.333333334))


# ## `np.linspace`
# For this reason, it is generally preferable to use the `linspace` function instead of `arange` when working with floats. The `linspace` function returns an array containing a specific number of points evenly distributed between two values (note that the maximum value is *included*, contrary to `arange`):

# In[19]:


print(np.linspace(0, 5/3, 6))


# ## `np.rand` and `np.randn`
# A number of functions are available in NumPy's `random` module to create `ndarray`s initialized with random values.
# For example, here is a 3x4 matrix initialized with random floats between 0 and 1 (uniform distribution):

# In[20]:


np.random.rand(3,4)


# Here's a 3x4 matrix containing random floats sampled from a univariate [normal distribution](https://en.wikipedia.org/wiki/Normal_distribution) (Gaussian distribution) of mean 0 and variance 1:

# In[21]:


np.random.randn(3,4)


# To give you a feel of what these distributions look like, let's use matplotlib (see the [matplotlib tutorial](tools_matplotlib.ipynb) for more details):

# In[22]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[23]:


plt.hist(np.random.rand(100000), normed=True, bins=100, histtype="step", color="blue", label="rand")
plt.hist(np.random.randn(100000), normed=True, bins=100, histtype="step", color="red", label="randn")
plt.axis([-2.5, 2.5, 0, 1.1])
plt.legend(loc = "upper left")
plt.title("Random distributions")
plt.xlabel("Value")
plt.ylabel("Density")
plt.show()


# ## np.fromfunction
# You can also initialize an `ndarray` using a function:

# In[24]:


def my_function(z, y, x):
    return x * y + z

np.fromfunction(my_function, (3, 2, 10))


# NumPy first creates three `ndarrays` (one per dimension), each of shape `(2, 10)`. Each array has values equal to the coordinate along a specific axis. For example, all elements in the `z` array are equal to their z-coordinate:
# 
#     [[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
#       [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]]
#     
#      [[ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
#       [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]]
#     
#      [[ 2.  2.  2.  2.  2.  2.  2.  2.  2.  2.]
#       [ 2.  2.  2.  2.  2.  2.  2.  2.  2.  2.]]]
# 
# So the terms x, y and z in the expression `x * y + z` above are in fact `ndarray`s (we will discuss arithmetic operations on arrays below).  The point is that the function `my_function` is only called *once*, instead of once per element. This makes initialization very efficient.

# # Array data
# ## `dtype`
# NumPy's `ndarray`s are also efficient in part because all their elements must have the same type (usually numbers).
# You can check what the data type is by looking at the `dtype` attribute:

# In[25]:


c = np.arange(1, 5)
print(c.dtype, c)


# In[26]:


c = np.arange(1.0, 5.0)
print(c.dtype, c)


# Instead of letting NumPy guess what data type to use, you can set it explicitly when creating an array by setting the `dtype` parameter:

# In[27]:


d = np.arange(1, 5, dtype=np.complex64)
print(d.dtype, d)


# Available data types include `int8`, `int16`, `int32`, `int64`, `uint8`|`16`|`32`|`64`, `float16`|`32`|`64` and `complex64`|`128`. Check out [the documentation](http://docs.scipy.org/doc/numpy-1.10.1/user/basics.types.html) for the full list.
# 
# ## `itemsize`
# The `itemsize` attribute returns the size (in bytes) of each item:

# In[28]:


e = np.arange(1, 5, dtype=np.complex64)
e.itemsize


# ## `data` buffer
# An array's data is actually stored in memory as a flat (one dimensional) byte buffer. It is available *via* the `data` attribute (you will rarely need it, though).

# In[29]:


f = np.array([[1,2],[1000, 2000]], dtype=np.int32)
f.data


# In python 2, `f.data` is a buffer. In python 3, it is a memoryview.

# In[30]:


if (hasattr(f.data, "tobytes")):
    data_bytes = f.data.tobytes() # python 3
else:
    data_bytes = memoryview(f.data).tobytes() # python 2

data_bytes


# Several `ndarrays` can share the same data buffer, meaning that modifying one will also modify the others. We will see an example in a minute.

# # Reshaping an array
# ## In place
# Changing the shape of an `ndarray` is as simple as setting its `shape` attribute. However, the array's size must remain the same.

# In[31]:


g = np.arange(24)
print(g)
print("Rank:", g.ndim)


# In[32]:


g.shape = (6, 4)
print(g)
print("Rank:", g.ndim)


# In[33]:


g.shape = (2, 3, 4)
print(g)
print("Rank:", g.ndim)


# ## `reshape`
# The `reshape` function returns a new `ndarray` object pointing at the *same* data. This means that modifying one array will also modify the other.

# In[34]:


g2 = g.reshape(4,6)
print(g2)
print("Rank:", g2.ndim)


# Set item at row 1, col 2 to 999 (more about indexing below).

# In[35]:


g2[1, 2] = 999
g2


# The corresponding element in `g` has been modified.

# In[36]:


g


# ## `ravel`
# Finally, the `ravel` function returns a new one-dimensional `ndarray` that also points to the same data:

# In[37]:


g.ravel()


# # Arithmetic operations
# All the usual arithmetic operators (`+`, `-`, `*`, `/`, `//`, `**`, etc.) can be used with `ndarray`s. They apply *elementwise*:

# In[38]:


a = np.array([14, 23, 32, 41])
b = np.array([5,  4,  3,  2])
print("a + b  =", a + b)
print("a - b  =", a - b)
print("a * b  =", a * b)
print("a / b  =", a / b)
print("a // b  =", a // b)
print("a % b  =", a % b)
print("a ** b =", a ** b)


# Note that the multiplication is *not* a matrix multiplication. We will discuss matrix operations below.
# 
# The arrays must have the same shape. If they do not, NumPy will apply the *broadcasting rules*.

# # Broadcasting

# In general, when NumPy expects arrays of the same shape but finds that this is not the case, it applies the so-called *broadcasting* rules:
# 
# ## First rule
# *If the arrays do not have the same rank, then a 1 will be prepended to the smaller ranking arrays until their ranks match.*

# In[39]:


h = np.arange(5).reshape(1, 1, 5)
h


# Now let's try to add a 1D array of shape `(5,)` to this 3D array of shape `(1,1,5)`. Applying the first rule of broadcasting!

# In[40]:


h + [10, 20, 30, 40, 50]  # same as: h + [[[10, 20, 30, 40, 50]]]


# ## Second rule
# *Arrays with a 1 along a particular dimension act as if they had the size of the array with the largest shape along that dimension. The value of the array element is repeated along that dimension.*

# In[41]:


k = np.arange(6).reshape(2, 3)
k


# Let's try to add a 2D array of shape `(2,1)` to this 2D `ndarray` of shape `(2, 3)`. NumPy will apply the second rule of broadcasting:

# In[42]:


k + [[100], [200]]  # same as: k + [[100, 100, 100], [200, 200, 200]]


# Combining rules 1 & 2, we can do this:

# In[43]:


k + [100, 200, 300]  # after rule 1: [[100, 200, 300]], and after rule 2: [[100, 200, 300], [100, 200, 300]]


# And also, very simply:

# In[44]:


k + 1000  # same as: k + [[1000, 1000, 1000], [1000, 1000, 1000]]


# ## Third rule
# *After rules 1 & 2, the sizes of all arrays must match.*

# In[45]:


try:
    k + [33, 44]
except ValueError as e:
    print(e)


# Broadcasting rules are used in many NumPy operations, not just arithmetic operations, as we will see below.
# For more details about broadcasting, check out [the documentation](https://docs.scipy.org/doc/numpy-dev/user/basics.broadcasting.html).

# ## Upcasting
# When trying to combine arrays with different `dtype`s, NumPy will *upcast* to a type capable of handling all possible values (regardless of what the *actual* values are).

# In[46]:


k1 = np.arange(0, 5, dtype=np.uint8)
print(k1.dtype, k1)


# In[47]:


k2 = k1 + np.array([5, 6, 7, 8, 9], dtype=np.int8)
print(k2.dtype, k2)


# Note that `int16` is required to represent all *possible* `int8` and `uint8` values (from -128 to 255), even though in this case a uint8 would have sufficed.

# In[48]:


k3 = k1 + 1.5
print(k3.dtype, k3)


# # Conditional operators

# The conditional operators also apply elementwise:

# In[49]:


m = np.array([20, -5, 30, 40])
m < [15, 16, 35, 36]


# And using broadcasting:

# In[50]:


m < 25  # equivalent to m < [25, 25, 25, 25]


# This is most useful in conjunction with boolean indexing (discussed below).

# In[51]:


m[m < 25]


# # Mathematical and statistical functions

# Many mathematical and statistical functions are available for `ndarray`s.
# 
# ## `ndarray` methods
# Some functions are simply `ndarray` methods, for example:

# In[52]:


a = np.array([[-2.5, 3.1, 7], [10, 11, 12]])
print(a)
print("mean =", a.mean())


# Note that this computes the mean of all elements in the `ndarray`, regardless of its shape.
# 
# Here are a few more useful `ndarray` methods:

# In[53]:


for func in (a.min, a.max, a.sum, a.prod, a.std, a.var):
    print(func.__name__, "=", func())


# These functions accept an optional argument `axis` which lets you ask for the operation to be performed on elements along the given axis. For example:

# In[54]:


c=np.arange(24).reshape(2,3,4)
c


# In[55]:


c.sum(axis=0)  # sum across matrices


# In[56]:


c.sum(axis=1)  # sum across rows


# You can also sum over multiple axes:

# In[57]:


c.sum(axis=(0,2))  # sum across matrices and columns


# In[58]:


0+1+2+3 + 12+13+14+15, 4+5+6+7 + 16+17+18+19, 8+9+10+11 + 20+21+22+23


# ## Universal functions
# NumPy also provides fast elementwise functions called *universal functions*, or **ufunc**. They are vectorized wrappers of simple functions. For example `square` returns a new `ndarray` which is a copy of the original `ndarray` except that each element is squared:

# In[59]:


a = np.array([[-2.5, 3.1, 7], [10, 11, 12]])
np.square(a)


# Here are a few more useful unary ufuncs:

# In[60]:


print("Original ndarray")
print(a)
for func in (np.abs, np.sqrt, np.exp, np.log, np.sign, np.ceil, np.modf, np.isnan, np.cos):
    print("\n", func.__name__)
    print(func(a))


# ## Binary ufuncs
# There are also many binary ufuncs, that apply elementwise on two `ndarray`s.  Broadcasting rules are applied if the arrays do not have the same shape:

# In[61]:


a = np.array([1, -2, 3, 4])
b = np.array([2, 8, -1, 7])
np.add(a, b)  # equivalent to a + b


# In[62]:


np.greater(a, b)  # equivalent to a > b


# In[63]:


np.maximum(a, b)


# In[64]:


np.copysign(a, b)


# # Array indexing
# ## One-dimensional arrays
# One-dimensional NumPy arrays can be accessed more or less like regular python arrays:

# In[65]:


a = np.array([1, 5, 3, 19, 13, 7, 3])
a[3]


# In[66]:


a[2:5]


# In[67]:


a[2:-1]


# In[68]:


a[:2]


# In[69]:


a[2::2]


# In[70]:


a[::-1]


# Of course, you can modify elements:

# In[71]:


a[3]=999
a


# You can also modify an `ndarray` slice:

# In[72]:


a[2:5] = [997, 998, 999]
a


# ## Differences with regular python arrays
# Contrary to regular python arrays, if you assign a single value to an `ndarray` slice, it is copied across the whole slice, thanks to broadcasting rules discussed above.

# In[73]:


a[2:5] = -1
a


# Also, you cannot grow or shrink `ndarray`s this way:

# In[74]:


try:
    a[2:5] = [1,2,3,4,5,6]  # too long
except ValueError as e:
    print(e)


# You cannot delete elements either:

# In[75]:


try:
    del a[2:5]
except ValueError as e:
    print(e)


# Last but not least, `ndarray` **slices are actually *views*** on the same data buffer. This means that if you create a slice and modify it, you are actually going to modify the original `ndarray` as well!

# In[76]:


a_slice = a[2:6]
a_slice[1] = 1000
a  # the original array was modified!


# In[77]:


a[3] = 2000
a_slice  # similarly, modifying the original array modifies the slice!


# If you want a copy of the data, you need to use the `copy` method:

# In[78]:


another_slice = a[2:6].copy()
another_slice[1] = 3000
a  # the original array is untouched


# In[79]:


a[3] = 4000
another_slice  # similary, modifying the original array does not affect the slice copy


# ## Multi-dimensional arrays
# Multi-dimensional arrays can be accessed in a similar way by providing an index or slice for each axis, separated by commas:

# In[80]:


b = np.arange(48).reshape(4, 12)
b


# In[81]:


b[1, 2]  # row 1, col 2


# In[82]:


b[1, :]  # row 1, all columns


# In[83]:


b[:, 1]  # all rows, column 1


# **Caution**: note the subtle difference between these two expressions: 

# In[84]:


b[1, :]


# In[85]:


b[1:2, :]


# The first expression returns row 1 as a 1D array of shape `(12,)`, while the second returns that same row as a 2D array of shape `(1, 12)`.

# ## Fancy indexing
# You may also specify a list of indices that you are interested in. This is referred to as *fancy indexing*.

# In[86]:


b[(0,2), 2:5]  # rows 0 and 2, columns 2 to 4 (5-1)


# In[87]:


b[:, (-1, 2, -1)]  # all rows, columns -1 (last), 2 and -1 (again, and in this order)


# If you provide multiple index arrays, you get a 1D `ndarray` containing the values of the elements at the specified coordinates.

# In[88]:


b[(-1, 2, -1, 2), (5, 9, 1, 9)]  # returns a 1D array with b[-1, 5], b[2, 9], b[-1, 1] and b[2, 9] (again)


# ## Higher dimensions
# Everything works just as well with higher dimensional arrays, but it's useful to look at a few examples:

# In[89]:


c = b.reshape(4,2,6)
c


# In[90]:


c[2, 1, 4]  # matrix 2, row 1, col 4


# In[91]:


c[2, :, 3]  # matrix 2, all rows, col 3


# If you omit coordinates for some axes, then all elements in these axes are returned:

# In[92]:


c[2, 1]  # Return matrix 2, row 1, all columns.  This is equivalent to c[2, 1, :]


# ## Ellipsis (`...`)
# You may also write an ellipsis (`...`) to ask that all non-specified axes be entirely included.

# In[93]:


c[2, ...]  #  matrix 2, all rows, all columns.  This is equivalent to c[2, :, :]


# In[94]:


c[2, 1, ...]  # matrix 2, row 1, all columns.  This is equivalent to c[2, 1, :]


# In[95]:


c[2, ..., 3]  # matrix 2, all rows, column 3.  This is equivalent to c[2, :, 3]


# In[96]:


c[..., 3]  # all matrices, all rows, column 3.  This is equivalent to c[:, :, 3]


# ## Boolean indexing
# You can also provide an `ndarray` of boolean values on one axis to specify the indices that you want to access.

# In[97]:


b = np.arange(48).reshape(4, 12)
b


# In[98]:


rows_on = np.array([True, False, True, False])
b[rows_on, :]  # Rows 0 and 2, all columns. Equivalent to b[(0, 2), :]


# In[99]:


cols_on = np.array([False, True, False] * 4)
b[:, cols_on]  # All rows, columns 1, 4, 7 and 10


# ## `np.ix_`
# You cannot use boolean indexing this way on multiple axes, but you can work around this by using the `ix_` function:

# In[100]:


b[np.ix_(rows_on, cols_on)]


# In[101]:


np.ix_(rows_on, cols_on)


# If you use a boolean array that has the same shape as the `ndarray`, then you get in return a 1D array containing all the values that have `True` at their coordinate. This is generally used along with conditional operators:

# In[102]:


b[b % 3 == 1]


# # Iterating
# Iterating over `ndarray`s is very similar to iterating over regular python arrays. Note that iterating over multidimensional arrays is done with respect to the first axis.

# In[103]:


c = np.arange(24).reshape(2, 3, 4)  # A 3D array (composed of two 3x4 matrices)
c


# In[104]:


for m in c:
    print("Item:")
    print(m)


# In[105]:


for i in range(len(c)):  # Note that len(c) == c.shape[0]
    print("Item:")
    print(c[i])


# If you want to iterate on *all* elements in the `ndarray`, simply iterate over the `flat` attribute:

# In[106]:


for i in c.flat:
    print("Item:", i)


# # Stacking arrays
# It is often useful to stack together different arrays. NumPy offers several functions to do just that. Let's start by creating a few arrays.

# In[107]:


q1 = np.full((3,4), 1.0)
q1


# In[108]:


q2 = np.full((4,4), 2.0)
q2


# In[109]:


q3 = np.full((3,4), 3.0)
q3


# ## `vstack`
# Now let's stack them vertically using `vstack`:

# In[110]:


q4 = np.vstack((q1, q2, q3))
q4


# In[111]:


q4.shape


# This was possible because q1, q2 and q3 all have the same shape (except for the vertical axis, but that's ok since we are stacking on that axis).
# 
# ## `hstack`
# We can also stack arrays horizontally using `hstack`:

# In[112]:


q5 = np.hstack((q1, q3))
q5


# In[113]:


q5.shape


# This is possible because q1 and q3 both have 3 rows. But since q2 has 4 rows, it cannot be stacked horizontally with q1 and q3:

# In[114]:


try:
    q5 = np.hstack((q1, q2, q3))
except ValueError as e:
    print(e)


# ## `concatenate`
# The `concatenate` function stacks arrays along any given existing axis.

# In[115]:


q7 = np.concatenate((q1, q2, q3), axis=0)  # Equivalent to vstack
q7


# In[116]:


q7.shape


# As you might guess, `hstack` is equivalent to calling `concatenate` with `axis=1`.

# ## `stack`
# The `stack` function stacks arrays along a new axis. All arrays have to have the same shape.

# In[117]:


q8 = np.stack((q1, q3))
q8


# In[118]:


q8.shape


# # Splitting arrays
# Splitting is the opposite of stacking. For example, let's use the `vsplit` function to split a matrix vertically.
# 
# First let's create a 6x4 matrix:

# In[119]:


r = np.arange(24).reshape(6,4)
r


# Now let's split it in three equal parts, vertically:

# In[120]:


r1, r2, r3 = np.vsplit(r, 3)
r1


# In[121]:


r2


# In[122]:


r3


# There is also a `split` function which splits an array along any given axis. Calling `vsplit` is equivalent to calling `split` with `axis=0`. There is also an `hsplit` function, equivalent to calling `split` with `axis=1`:

# In[123]:


r4, r5 = np.hsplit(r, 2)
r4


# In[124]:


r5


# # Transposing arrays
# The `transpose` method creates a new view on an `ndarray`'s data, with axes permuted in the given order.
# 
# For example, let's create a 3D array:

# In[125]:


t = np.arange(24).reshape(4,2,3)
t


# Now let's create an `ndarray` such that the axes `0, 1, 2` (depth, height, width) are re-ordered to `1, 2, 0` (depth→width, height→depth, width→height):

# In[126]:


t1 = t.transpose((1,2,0))
t1


# In[127]:


t1.shape


# By default, `transpose` reverses the order of the dimensions:

# In[128]:


t2 = t.transpose()  # equivalent to t.transpose((2, 1, 0))
t2


# In[129]:


t2.shape


# NumPy provides a convenience function `swapaxes` to swap two axes. For example, let's create a new view of `t` with depth and height swapped:

# In[130]:


t3 = t.swapaxes(0,1)  # equivalent to t.transpose((1, 0, 2))
t3


# In[131]:


t3.shape


# # Linear algebra
# NumPy 2D arrays can be used to represent matrices efficiently in python. We will just quickly go through some of the main matrix operations available. For more details about Linear Algebra, vectors and matrics, go through the [Linear Algebra tutorial](math_linear_algebra.ipynb).
# 
# ## Matrix transpose
# The `T` attribute is equivalent to calling `transpose()` when the rank is ≥2:

# In[132]:


m1 = np.arange(10).reshape(2,5)
m1


# In[133]:


m1.T


# The `T` attribute has no effect on rank 0 (empty) or rank 1 arrays:

# In[134]:


m2 = np.arange(5)
m2


# In[135]:


m2.T


# We can get the desired transposition by first reshaping the 1D array to a single-row matrix (2D):

# In[136]:


m2r = m2.reshape(1,5)
m2r


# In[137]:


m2r.T


# ## Matrix dot product
# Let's create two matrices and execute a matrix [dot product](https://en.wikipedia.org/wiki/Dot_product) using the `dot` method.

# In[138]:


n1 = np.arange(10).reshape(2, 5)
n1


# In[139]:


n2 = np.arange(15).reshape(5,3)
n2


# In[140]:


n1.dot(n2)


# **Caution**: as mentionned previously, `n1*n2` is *not* a dot product, it is an elementwise product.

# ## Matrix inverse and pseudo-inverse
# Many of the linear algebra functions are available in the `numpy.linalg` module, in particular the `inv` function to compute a square matrix's inverse:

# In[141]:


import numpy.linalg as linalg

m3 = np.array([[1,2,3],[5,7,11],[21,29,31]])
m3


# In[142]:


linalg.inv(m3)


# You can also compute the [pseudoinverse](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_pseudoinverse) using `pinv`:

# In[143]:


linalg.pinv(m3)


# ## Identity matrix
# The product of a matrix by its inverse returns the identiy matrix (with small floating point errors):

# In[144]:


m3.dot(linalg.inv(m3))


# You can create an identity matrix of size NxN by calling `eye`:

# In[145]:


np.eye(3)


# ## QR decomposition
# The `qr` function computes the [QR decomposition](https://en.wikipedia.org/wiki/QR_decomposition) of a matrix:

# In[146]:


q, r = linalg.qr(m3)
q


# In[147]:


r


# In[148]:


q.dot(r)  # q.r equals m3


# ## Determinant
# The `det` function computes the [matrix determinant](https://en.wikipedia.org/wiki/Determinant):

# In[149]:


linalg.det(m3)  # Computes the matrix determinant


# ## Eigenvalues and eigenvectors
# The `eig` function computes the [eigenvalues and eigenvectors](https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors) of a square matrix:

# In[150]:


eigenvalues, eigenvectors = linalg.eig(m3)
eigenvalues # λ


# In[151]:


eigenvectors # v


# In[152]:


m3.dot(eigenvectors) - eigenvalues * eigenvectors  # m3.v - λ*v = 0


# ## Singular Value Decomposition
# The `svd` function takes a matrix and returns its [singular value decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition):

# In[153]:


m4 = np.array([[1,0,0,0,2], [0,0,3,0,0], [0,0,0,0,0], [0,2,0,0,0]])
m4


# In[154]:


U, S_diag, V = linalg.svd(m4)
U


# In[155]:


S_diag


# The `svd` function just returns the values in the diagonal of Σ, but we want the full Σ matrix, so let's create it:

# In[156]:


S = np.zeros((4, 5))
S[np.diag_indices(4)] = S_diag
S  # Σ


# In[157]:


V


# In[158]:


U.dot(S).dot(V) # U.Σ.V == m4


# ## Diagonal and trace

# In[159]:


np.diag(m3)  # the values in the diagonal of m3 (top left to bottom right)


# In[160]:


np.trace(m3)  # equivalent to np.diag(m3).sum()


# ## Solving a system of linear scalar equations

# The `solve` function solves a system of linear scalar equations, such as:
# 
# * $2x + 6y = 6$
# * $5x + 3y = -9$

# In[161]:


coeffs  = np.array([[2, 6], [5, 3]])
depvars = np.array([6, -9])
solution = linalg.solve(coeffs, depvars)
solution


# Let's check the solution:

# In[162]:


coeffs.dot(solution), depvars  # yep, it's the same


# Looks good! Another way to check the solution:

# In[163]:


np.allclose(coeffs.dot(solution), depvars)


# # Vectorization
# Instead of executing operations on individual array items, one at a time, your code is much more efficient if you try to stick to array operations. This is called *vectorization*. This way, you can benefit from NumPy's many optimizations.
# 
# For example, let's say we want to generate a 768x1024 array based on the formula $sin(xy/40.5)$. A **bad** option would be to do the math in python using nested loops:

# In[164]:


import math
data = np.empty((768, 1024))
for y in range(768):
    for x in range(1024):
        data[y, x] = math.sin(x*y/40.5)  # BAD! Very inefficient.


# Sure, this works, but it's terribly inefficient since the loops are taking place in pure python. Let's vectorize this algorithm. First, we will use NumPy's `meshgrid` function which generates coordinate matrices from coordinate vectors.

# In[165]:


x_coords = np.arange(0, 1024)  # [0, 1, 2, ..., 1023]
y_coords = np.arange(0, 768)   # [0, 1, 2, ..., 767]
X, Y = np.meshgrid(x_coords, y_coords)
X


# In[166]:


Y


# As you can see, both `X` and `Y` are 768x1024 arrays, and all values in `X` correspond to the horizontal coordinate, while all values in `Y` correspond to the the vertical coordinate.
# 
# Now we can simply compute the result using array operations:

# In[167]:


data = np.sin(X*Y/40.5)


# Now we can plot this data using matplotlib's `imshow` function (see the [matplotlib tutorial](tools_matplotlib.ipynb)).

# In[168]:


import matplotlib.pyplot as plt
import matplotlib.cm as cm
fig = plt.figure(1, figsize=(7, 6))
plt.imshow(data, cmap=cm.hot, interpolation="bicubic")
plt.show()


# # Saving and loading
# NumPy makes it easy to save and load `ndarray`s in binary or text format.
# 
# ## Binary `.npy` format
# Let's create a random array and save it.

# In[169]:


a = np.random.rand(2,3)
a


# In[170]:


np.save("my_array", a)


# Done! Since the file name contains no file extension was provided, NumPy automatically added `.npy`. Let's take a peek at the file content:

# In[171]:


with open("my_array.npy", "rb") as f:
    content = f.read()

content


# To load this file into a NumPy array, simply call `load`:

# In[172]:


a_loaded = np.load("my_array.npy")
a_loaded


# ## Text format
# Let's try saving the array in text format:

# In[173]:


np.savetxt("my_array.csv", a)


# Now let's look at the file content:

# In[174]:


with open("my_array.csv", "rt") as f:
    print(f.read())


# This is a CSV file with tabs as delimiters. You can set a different delimiter:

# In[175]:


np.savetxt("my_array.csv", a, delimiter=",")


# To load this file, just use `loadtxt`:

# In[176]:


a_loaded = np.loadtxt("my_array.csv", delimiter=",")
a_loaded


# ## Zipped `.npz` format
# It is also possible to save multiple arrays in one zipped file:

# In[177]:


b = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
b


# In[178]:


np.savez("my_arrays", my_a=a, my_b=b)


# Again, let's take a peek at the file content. Note that the `.npz` file extension was automatically added.

# In[179]:


with open("my_arrays.npz", "rb") as f:
    content = f.read()

repr(content)[:180] + "[...]"


# You then load this file like so:

# In[180]:


my_arrays = np.load("my_arrays.npz")
my_arrays


# This is a dict-like object which loads the arrays lazily:

# In[181]:


my_arrays.keys()


# In[182]:


my_arrays["my_a"]


# # What next?
# Now you know all the fundamentals of NumPy, but there are many more options available. The best way to learn more is to experiment with NumPy, and go through the excellent [reference documentation](http://docs.scipy.org/doc/numpy/reference/index.html) to find more functions and features you may be interested in.
