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
