import tensorflow as tf
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


np.random.randint(0,10,size=10)

def ismember(a, b):
    rows_close = np.all((a - b[:, None]) == 0, axis=-1)
    return np.any(rows_close)

A = [1,2,3,4,5]
B = [2,3,4,5,6]
C = [A, B]
D = [B, A]

a = [1]
if a:
    print('true')


ismember(C,D)

#######################################################################################

mnist = tf.keras.datasets.mnist

# ipython的に
sess = tf.InteractiveSession()

x = tf.placeholder('float32', shape=[None, 784])
u = tf.placeholder('float32', shape=[None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
cross_entropy = -tf.reduce_sum(u * tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

sess.run(tf.initialize_all_variables())
print(sess.run(W))


for i in range(1000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})




#####################################################################################

# edgelistから，隣接行列，次数行列を作成
G = nx.DiGraph()
G.add_edge(1,2)
G.add_edge(1,3)
G.add_edge(1,4)
G.add_edge(1,5)
G.add_edge(2,3)
G.add_edge(2,4)
G.add_edge(3,5)
G.add_edge(3,6)
G.add_edge(6,7)
G.add_edge(6,8)
G.add_edge(7,8)
G.add_edge(7,9)

plt.figure()
pos = nx.spring_layout(G)
nx.draw_networkx(G,pos)
plt.show()

adj = nx.adjacency_matrix(G)
adj
adj.transpose().todense()
adj.todense()
def convert_sparse_matrix_to_sparse_tensor(X_sp):
    """
    scipyのsparse_matrixを，tfのsparse_tensorに変換
    tensorflowのsparse表現は，tf.SparseTensor(値を取るindicesのリストのリスト, 各値のリスト, dense_shape)を取る
    """
    coo = X_sp.tocoo()  # COO形式に
    indices = np.mat([coo.row, coo.col]).transpose()  # 各行が，行列の中で値を持っている成分の[行番号, 列番号]
    return tf.SparseTensor(indices, coo.data, coo.shape)

A_sp = convert_sparse_matrix_to_sparse_tensor(adj)
a = A_sp.indices.shape[0].value
type(a)
int(a)

num_nodes=5
sp.sparse.identity(num_nodes).todense()
sp.sparse.identity(num_nodes)





C1 = tf.constant([[1,2,3],[4,5,6],[7,8,9]])
C2 = tf.constant([[4,5,6],[7,8,9],[1,2,3]])
C3 = tf.constant([[7,8,9],[1,2,3],[4,5,6]])
a = tf.Variable([2,2,2])

sess.run(tf.global_variables_initializer())
a[0][0].eval()

(a[0]*C1).eval()


C4 = tf.mul()




# (N,N)*(N,1)の要素積は？
Z=tf.constant([[1,2,3],[4,5,6],[7,8,9]])
Y1=tf.constant([[1,2,3]])
Y2=tf.constant([[1],[2],[3]])
Z.eval()
Y1.eval()
Y2.eval()

(Z*Y1).eval()
(Z*Y2).eval()



sess.run(tf.shape(A_sp)[0])


sess.run(tf.random_uniform([10]))


adj.data
adj.shape
adj.tocoo().row
adj.tocoo().col

def sparse_to_tuple(sparse_mat):
    """
    sp.sparse型を，tf.SparseTensor型にするために，coords, values, shapeを作成(引数1つで取ると，tupleになる)
    coords: 各行に，行列のノンゼロとなる要素番号を取るnp.array（i.e. ノンゼロ個数*2の行列）
    values: ノンゼロとなる要素が持つ値のリスト（i.e. ノンゼロの数の長さのリスト）
    shape: 元の行列のshape
    """
    if not sp.sparse.isspmatrix_coo(sparse_mat):
        sparse_mat = sparse_mat.tocoo()  # COO形式に
    coords = np.vstack((sparse_mat.row, sparse_mat.col)).transpose()  # ノンゼロ行番号のリストと，ノンゼロ列番号のリストをstack
    values = sparse_mat.data
    shape = sparse_mat.shape
    return coords, values, shape

features = sparse_to_tuple(adj)
features
num_features = features[2][1]
features_nonzero = features[1].shape[0]
num_features
features_nonzero


np.vstack((1,2))

import tensorflow as tf
noise_shape=[10]
random_tensor = 0.5
random_tensor = 0
random_tensor += tf.random_uniform(noise_shape)
dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
pre_out = tf.sparse_retain(A_sp, dropout_mask)

sess = tf.InteractiveSession()
sess.run(random_tensor)




adj.todense()

A = tf.constant(adj.todense())
sess.run(A)

A = nx.to_scipy_sparse_matrix(G, format='coo')
A
B = A @ A

A.todense()
D = G.degree()
D
len(D)
D[1]
[i for i in range(1,len(D))]
1/D[1]

S = sp.sparse.dok_matrix((len(D), len(D)), dtype=np.float32)
for i in range(len(D)):
    S[i,i] = 1/D[i+1]
S
S.todense()


#############################################################################################################

def convert_sparse_matrix_to_sparse_tensor(X_sp):
    """
    scipyのsparse_matrixを，tfのsparse_tensorに変換
    tensorflowのsparse表現は，tf.SparseTensor(値を取るindicesのリストのリスト, 各値のリスト, dense_shape)を取る
    """
    coo = X_sp.tocoo()  # COO形式に
    indices = np.mat([coo.row, coo.col]).transpose()  # 各行が，行列の中で値を持っている成分の[行番号, 列番号]
    return tf.SparseTensor(indices, coo.data, coo.shape)


sess = tf.InteractiveSession()
A = adj
A_sp = convert_sparse_matrix_to_sparse_tensor(A)
D_sp = convert_sparse_matrix_to_sparse_tensor(S)
A_sp
D_sp

D_sp.indices[1].eval()
D_sp.indices[1][0].eval()
D_sp.indices[1][1].eval()

D_sp.indices.shape
int(D_sp.indices.shape[0]) +1
D_sp.indices[0].eval()

D_sp.indices.shape
[i for i in range(D_sp.indices.shape[0])]


D_sp.indices


D_sp.indices.shape[0]
A[D_sp.indices[0][0]].eval()


A = tf.constant([[1],[2],[3]])
B = tf.constant([[1],[2],[3]])
A.eval()
(A[0] + B[1])[0].eval()


aaa = tf.SparseTensor(indices=self.D_sp.indices, values=[tf.nn.leaky_relu(att_self[self.adj.indices[i][0]]+att_neigh[self.adj.indices[i][1]]) for n in range(int(self.adj.indices.shape[0]))], dense_shape=self.adj.dense_shape)



list(D_sp.indices).eval()

A_sp.eval()
D_sp.eval()
D = tf.constant([[1,2,3,4,5,6,7,8,9]], dtype=tf.int64)
cc = tf.sparse_tensor_dense_matmul(A_sp, tf.transpose(D))
cc.eval()

c = tf.sparse_tensor_dense_matmul(A_sp, D_sp)

c = tf.sparse_matmul(A_sp, D_sp, a_is_sparse=True, b_is_sparse=True)


c = tf.matmul(tf.sparse_tensor_to_dense(A_sp, 0), tf.sparse_tensor_to_dense(D_sp, 0), a_is_sparse=True, b_is_sparse=True)


tf.sparse_tensor_to_dense(A_sp, 0)
tf.sparse_tensor_to_dense(D_sp, 0)

sess.run(A_sp.dense_shape)
sess.run(D_sp.dense_shape)

len(G.nodes())

dim_in = 8
dim_out = 5
W = tf.constant(0.1, shape=[dim_in, dim_out], dtype=tf.float32)
sess.run(tf.shape(W))
#sess.run(W.get_shape())
#W = tf.constant(tf.random_uniform(shape=[len(G.nodes()), 5]), shape=[len(G.nodes()), 5])

X_in = tf.constant(0.1, shape=[len(G.nodes()), dim_in], dtype=tf.float32)

sess.run(A_sp)
sess.run(tf.shape(A_sp)[0])

sess.run(tf.sparse_tensor_to_dense(A_sp))

X_t = X_in @ W
sess.run(X_t)

X_temp = tf.sparse_tensor_dense_matmul(A_sp, X_t)

X_temp = tf.sparse_tensor_dense_matmul(D_sp, tf.sparse_tensor_to_dense(A_sp))

X_temp = D_sp @ A_sp @ X_in @ W

sess.run(W)

sess.run(tf.sqrt(W))
0.316**2

#############################################################################################################

H1 = tf.random_normal(shape=(len(G.nodes()), dim_in))
H2 = tf.random_normal(shape=(len(G.nodes()), dim_in))
sess.run(H1)
sess.run(H2)
H1 = tf.constant([[1,2],[3,4]])
H2 = tf.constant([[4,5],[6,7]])
sess.run(H1)
sess.run(tf.shape(H2)[0])


sess.run(H2*1/2)

sess.run(tf.ones((1,3)))


sess.run([H1,H2])

sess.run(tf.reduce_mean([H1,H2]))
sess.run(tf.reduce_mean([H1,H2], axis=0))
sess.run(tf.reduce_mean([H1,H2], axis=1))



sess.run(tf.reduce_mean([H1,H2], keep_dims=True))

sess.run(tf.add(H1,H2))
sess.run(tf.reduce_mean(tf.add(H1,H2), axis=1))


a = [2,3,4,6,5]
sorted(a)





###################

A = tf.constant([[1,2,3]])
B = tf.constant([[1],[2],[3]])
sess.run(A)
sess.run(B)
sess.run(A+B)
sess.run(tf.add(A,B))
A.eval()

sp_mat = tf.SparseTensor([[0,0],[0,2],[1,2],[2,1]], np.ones(4), [3,3])
const1 = tf.constant([[1,2,3],[4,5,6],[7,8,9]], dtype=tf.float64)
const2 = tf.constant(np.array([1,2,3]),dtype=tf.float64)

elementwise_result = sp_mat.__mul__(const1)
broadcast_result   = sp_mat.__mul__(const2)

print("Sparse Matrix:\n",tf.sparse_tensor_to_dense(sp_mat).eval())
const1.eval()
print("\n\nElementwise:\n",tf.sparse_tensor_to_dense(elementwise_result).eval())
print("\n\nBroadcast:\n",tf.sparse_tensor_to_dense(broadcast_result).eval())


adj
adj.shape[0]


adj.todense()
adj.diagonal()
[adj.diagonal()[np.newaxis,:], [0]]

sp.sparse.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape).todense()


adj_orig = adj - sp.sparse.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
adj_orig.eliminate_zeros()







D = np.sum(A, axis=1)
D

np.diag([1,2,3])


coo = A.tocoo()
coo
A
A
sp.sparse.linalg.inv(A)


coo.row
len(coo.row)
coo.col
coo.data
coo.shape
A.todense()
np.mat([coo.row, coo.col])
indices = np.mat([coo.row, coo.col]).transpose()
indices
A_tf = tf.SparseTensor(indices, coo.data, coo.shape)

sess = tf.InteractiveSession()
sess.run(A_tf)


b = tf.Variable(0, shape=[dim_out, 1])
b = tf.Variable(tf.zeros([10]))
sess.run(b)


################################################################################

def weight_variable_glorot(input_dim, output_dim, name=""):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform(
        [input_dim, output_dim], minval=-init_range,
        maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)

input = weight_variable_glorot(10,5, name='aa')
w1 = weight_variable_glorot(5,4, name='aa')
w2 = weight_variable_glorot(4,1, name='bb')

sess.run(tf.global_variables_initializer())
input.eval()
w1.eval()
w2.eval()
A = tf.matmul(input, w1)
A = tf.matmul(A, w2)
A.eval()

tf.matmul(w1, w2).eval()
print(A.eval())

w.eval()
##############################################################################

import scipy.sparse as sp

def sparse_to_tuple(sparse_mx):
    """
    sp.sparse行列を，　値を持つcoords，　値の中身, 本来の行列の形のtupleに変換
    """
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

num_nodes =100
features = sparse_to_tuple(sp.identity(num_nodes))
features


A = ['a', 'b', 'aa']
sorted(A)
