# Glossary

**Broadcasting operation**

An operation that uses [numpy-style broadcasting](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
to make the shapes of its tensor arguments compatible.

**Device**

A piece of hardware that can run computation and has its own address space,
like a GPU or CPU.

**eval**

A method of `Tensor` that returns the value of the `Tensor`, triggering any
graph computation required to determine the value. You may only call `eval()`
on a `Tensor` in a graph that has been launched in a session.

**Feed**

TensorFlow's mechanism for patching a tensor directly into any node in a graph
launched in a session. You apply feeds when you trigger the execution of a
graph, not when you build the graph. A feed temporarily replaces a node with a
tensor value. You supply feed data as an argument to a `run()` or `eval()` call
that initiates computation. After the run the feed disappears and the original
node definition remains. You usually designate specific nodes to be "feed"
nodes by using `tf.placeholder()` to create them. See
[Basic Usage](../get_started/basic_usage.md) for more information.

**Fetch**

TensorFlow's mechanism for retrieving tensors from a graph launched in a
session. You retrieve fetches when you trigger the execution of a graph, not
when you build the graph. To fetch the tensor value of a node or nodes,
execute the graph with a `run()` call on the `Session` object and pass a list of
names of nodes to retrieve. See [Basic Usage](../get_started/basic_usage.md)
for more information.

**Graph**

Describes a computation as a directed acyclic
graph.  Nodes in the graph represent operations that must be
performed. Edges in the graph represent either data or control
dependencies. `GraphDef` is the proto used to describe a graph to the
system (it is the API), and consists of a collection of `NodeDefs` (see
below). A `GraphDef` may be converted to a (C++) `Graph` object which is
easier to operate on.

**IndexedSlices**

In the Python API, TensorFlow's representation of a tensor that is sparse
along only its first dimension. If the tensor is `k`-dimensional, an
`IndexedSlices` instance logically represents a collection of
`(k-1)`-dimensional slices along the tensor's first dimension. The indices of
the slices are stored concatenated into a single 1-dimensional vector, and the
corresponding slices are concatenated to form a single `k`-dimensional tensor. Use
`SparseTensor` if the sparsity is not restricted to the first dimension.

**Node**

An element of a graph.

Describes how to invoke a specific operation as one node in a specific
computation `Graph`, including the values for any `attrs` needed to configure
the operation. For operations that are polymorphic, the `attrs` include
sufficient information to completely determine the signature of the `Node`.
See `graph.proto` for details.

**Op (operation)**

In the TensorFlow runtime: A type of computation such as `add` or `matmul` or
`concat`.  You can add new ops to the runtime as described [how to add an
op](../how_tos/adding_an_op/index.md).

In the Python API: A node in the graph.  Ops are represented by instances of
the class [`tf.Operation`](../api_docs/python/framework.md#Operation).  The
`type` property of an `Operation` indicates the run operation for the node,
such as `add` or `matmul`.

**Run**

The action of executing ops in a launched graph.  Requires that the graph be
launched in a `Session`.

In the Python API: A method of the `Session` class:
[`tf.Session.run`](../api_docs/python/client.md#Session).  You can pass tensors
to feed and fetch to the `run()` call.

In the C++ API: A method of the [`tensorflow::Session`](../api_docs/cc/ClassSession.md).

**Session**

A runtime object representing a launched graph.  Provides methods to execute
ops in the graph.

In the Python API: [`tf.Session`](../api_docs/python/client.md#Session)

In the C++ API: class used to launch a graph and run operations
[`tensorflow::Session`](../api_docs/cc/ClassSession.md).

**Shape**

The number of dimensions of a tensor and their sizes.

In a launched graph: Property of the tensors that flow between nodes.  Some ops
have strong requirements on the shape of their inputs and report errors at
runtime if these are not met.

In the Python API: Attribute of a Python `Tensor` in the graph construction
API. During constructions the shape of tensors can be only partially known, or
even unknown.  See
[`tf.TensorShape`](../api_docs/python/framework.md#TensorShape)

In the C++ API: class used to represent the shape of tensors
[`tensorflow::TensorShape`](../api_docs/cc/ClassTensorShape.md).

**SparseTensor**

In the Python API, TensorFlow's representation of a tensor that is sparse in
arbitrary positions. A `SparseTensor` stores only the non-empty values along
with their indices, using a dictionary-of-keys format. In other words, if
there are `m` non-empty values, it maintains a length-`m` vector of values and
a matrix with m rows of indices. For efficiency, `SparseTensor` requires the
indices to be sorted along increasing dimension number, i.e. in row-major
order. Use `IndexedSlices` if the sparsity is only along the first dimension.

**Tensor**

A `Tensor` is a typed multi-dimensional array.  For example, a 4-D
array of floating point numbers representing a mini-batch of images with
dimensions `[batch, height, width, channel]`.

In a launched graph: Type of the data that flow between nodes.

In the Python API: class used to represent the output and inputs of ops added
to the graph [`tf.Tensor`](../api_docs/python/framework.md#Tensor).  Instances of
this class do not hold data.

In the C++ API: class used to represent tensors returned from a
[`Session::Run()`](../api_docs/cc/ClassSession.md) call
[`tensorflow::Tensor`](../api_docs/cc/ClassTensor.md).
Instances of this class hold data.
