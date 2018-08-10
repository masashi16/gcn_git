import numpy as np
import scipy as sp
import networkx as nx
import tensorflow as tf
import time


class GCNModel():
    def __init__(self, placeholders, num_features, features_nonzero, name):
        self.name = name
        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.vars = {}
        with tf.variable_scope(self.name):
            self.vars['attention_layer'] = tf.Variable([1,1,1], dtype=tf.float32, name="attention_layer")
            self.build()

    def build(self):
        self.hidden1 = GraphConvolutionSparse_Attention(
            name='gcn_sparse_layer',
            input_dim=self.input_dim,
            output_dim=FLAGS.hidden1,
            adj=self.adj,
            features_nonzero=self.features_nonzero,
            act=tf.nn.relu,
            dropout=self.dropout)(self.inputs)

        self.hidden2 = GraphConvolution_Attention(
            name='gcn_dense_layer_1',
            input_dim=FLAGS.hidden1,
            output_dim=FLAGS.hidden2,
            adj=self.adj,
            act=lambda x: x,
            dropout=self.dropout)(self.hidden1)

        self.hidden3 = GraphConvolution_Attention(
            name='gcn_dense_layer_2',
            input_dim=FLAGS.hidden2,
            output_dim=FLAGS.hidden3,
            adj=self.adj,
            act=lambda x: x,
            dropout=self.dropout)(self.hidden2)

        self.hidden4 = GraphConvolution_Attention(
            name='gcn_dense_layer_3',
            input_dim=FLAGS.hidden3,
            output_dim=FLAGS.hidden4,
            adj=self.adj,
            act=lambda x: x,
            dropout=self.dropout)(self.hidden3)

        # 層方向のattention-layer
        weight = tf.nn.softmax(self.vars['attention_layer'] )
        self.embeddings = self.vars['attention_layer'][0]*self.hidden2 + self.vars['attention_layer'][1]*self.hidden3 + self.vars['attention_layer'][2]*self.hidden4

        self.reconstructions = InnerProductDecoder(
            name='gcn_decoder',
            input_dim=FLAGS.hidden4,
            act=lambda x: x)(self.embeddings)


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
