import numpy as np
import scipy as sp
import networkx as nx
import tensorflow as tf
import time
import gc


class GraphConvolution():
    """Basic graph convolution layer for undirected graph without edge labels.
    __init__は，インスタンスが生成された時に自動的に実行されるのに対し，
    __call__は生成したインスタンスを関数のように呼び出した時に自動的に実行される
    """
    def __init__(self, input_dim, output_dim, adj, name, dropout=0., act=tf.nn.relu):
        self.name = name
        self.vars = {}
        self.issparse = False
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name='weights')
        self.dropout = dropout
        self.adj = adj
        self.act = act

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            X = inputs
            X = tf.nn.dropout(X, 1-self.dropout)
            X = tf.matmul(X, self.vars['weights'])
            X = tf.sparse_tensor_dense_matmul(self.adj, X)
            outputs = self.act(X)
        return outputs



class GraphConvolutionSparse():
    """Graph convolution layer for sparse inputs."""
    def __init__(self, input_dim, output_dim, adj, features_nonzero, name, dropout=0., act=tf.nn.relu):
        self.name = name
        self.vars = {}
        self.issparse = False
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name='weights')
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            X = inputs
            X = dropout_sparse(X, 1-self.dropout, self.features_nonzero)
            X = tf.sparse_tensor_dense_matmul(X, self.vars['weights'])
            X = tf.sparse_tensor_dense_matmul(self.adj, X)
            outputs = self.act(X)
        return outputs



class GraphConvolution_Attention():
    """
    attentionつき
    """
    def __init__(self, input_dim, output_dim, adj, name, dropout=0., act=tf.nn.relu):
        self.name = name
        self.vars = {}
        self.issparse = False
        self.dropout = dropout
        self.adj = adj
        self.act = act
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name='weights')
            self.vars['attention_self'] = weight_variable_glorot(output_dim, 1, name="attention_self")
            self.vars['attention_neigh'] = weight_variable_glorot(output_dim, 1, name="attention_neigh")


    def __call__(self, inputs):
        with tf.name_scope(self.name):
            x = inputs
            x = tf.nn.dropout(x, 1-self.dropout)
            x = tf.matmul(x, self.vars['weights'])

            # XWa = XWa_{self} + XWa_{neigh} と分ける
            att_self = tf.matmul(x, self.vars['attention_self'])  # (N,1)
            att_neigh = tf.matmul(x, self.vars['attention_neigh'])  # (N,1)
            att = tf.add(att_self, tf.transpose(att_neigh))  # (N,1) + (1,N)　→ (N,N)にbroadcastされるのを利用
            att = self.adj.__mul__(att)  # 隣接行列と要素積を取ったもの(__mul__でsparse型との要素積が可能, 結果はsparseとなる)
            att = tf.SparseTensor(indices=att.indices, values=tf.nn.leaky_relu(att.values, alpha=0.2), dense_shape=att.dense_shape)
            att = tf.sparse_softmax(att)

            x = tf.sparse_tensor_dense_matmul(att, x)
            #x = tf.matmul(att, x)
            outputs = self.act(x)
        return outputs


class GraphConvolutionSparse_Attention():
    """Graph convolution layer for sparse inputs."""
    def __init__(self, input_dim, output_dim, adj, features_nonzero, name, dropout=0., act=tf.nn.relu):
        self.name = name
        self.vars = {}
        self.issparse = False
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name='weights')
            self.vars['attention_self'] = weight_variable_glorot(output_dim, 1, name="attention_self")
            self.vars['attention_neigh'] = weight_variable_glorot(output_dim, 1, name="attention_neigh")

    #def attention(self, X):
        # XWの計算
        #X_linear_tr = tf.matmul(X, self.vars['weights'])
        # XWa = XWa_{self} + XWa_{neigh} と分ける
        #att_self = tf.matmul(X_linear_tr, self.vars['attention_self'])  # (N,1)
        #att_neigh = tf.matmul(X_linear_tr, self.vars['attention_neigh'])  # (N,1)
        #att = tf.add(att_self, tf.transpose(att_neigh))  # (N,1) + (1,N)　→ (N,N)にbroadcastされるのを利用

        #del X_linear_tr
        #gc.collect()

        #return self.adj.__mul__(att)  # 隣接行列と要素積を取ったもの(__mul__でsparse型との要素積が可能)

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            x = inputs
            x = dropout_sparse(x, 1-self.dropout, self.features_nonzero)
            x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])  # この結果は， denseとなる

            # XWa = XWa_{self} + XWa_{neigh} と分ける
            att_self = tf.matmul(x, self.vars['attention_self'])  # (N,1)
            att_neigh = tf.matmul(x, self.vars['attention_neigh'])  # (N,1)

            # SparseTensor型で，　attentionの重みを作成
            # 注意として，　placeholderは入るまで分からないため，　型とか値とかは，　この時点では，　？ なので，　例えば，　
            #  att = tf.SparseTensor(indices=self.adj.indices, values=[att_self[self.adj.indices[i][0]]+att_neigh[self.adj.indices[i][1]] for i in range(self.adj.indices[0].value)], dense_shape=self.adj.dense_shape)
            # 上記は，　self.adj.indices[0].valueがNoneゆえ，　ループが回らなく，　エラーとなる

            # (N,N)と(N,1)の要素積は，　(N,1)を(N,N)の列方向に各列に掛けていったものになる
            # (N,N)と(1,N)の要素積は，　(1,N)を(N,N)の行方向に各行に掛けていったものになる

            att_1 = self.adj.__mul__(tf.nn.leaky_relu(att_self, alpha=0.2))
            att_2 = self.adj.__mul__(tf.transpose(tf.nn.leaky_relu(att_neigh, alpha=0.2)))
            att = tf.sparse_add(att_1, att_2)

            del att_1,   att_2
            gc.collect()

            #att = tf.add(att_self, tf.transpose(att_neigh))  # (N,1) + (1,N)　→ (N,N)にbroadcastされるのを利用
            #att = self.adj.__mul__(att)  # 隣接行列と要素積を取ったもの(adjはsparseゆえ，　__mul__でsparse型との要素積， 結果はsparseとなる)
            #att = tf.SparseTensor(indices=att.indices, values=tf.nn.leaky_relu(att.values), dense_shape=att.dense_shape)

            att = tf.sparse_softmax(att)

            x = tf.sparse_tensor_dense_matmul(att, x)
            outputs = self.act(x)
        return outputs


class InnerProductDecoder():
    """Decoder model layer for link prediction."""
    def __init__(self, input_dim, name, dropout=0., act=tf.nn.sigmoid):
        self.name = name
        self.issparse = False
        self.dropout = dropout
        self.act = act

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            inputs = tf.nn.dropout(inputs, 1-self.dropout)
            X = tf.transpose(inputs)
            X = tf.matmul(inputs, X)
            X = tf.reshape(X, [-1])
            outputs = self.act(X)
        return outputs
