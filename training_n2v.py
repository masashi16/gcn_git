import argparse
from time import time
import numpy as np
import networkx as nx
import node2vec
from gensim.models import Word2Vec
from glob import glob
import gc


def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--input', nargs='?', default='data/karate.edgelist',
                        help='Input graph path. Graph data must be edgelist.')
    parser.add_argument('--output_model', nargs='?', default='emb/karate.model',
                        help='Embedding model path')
    parser.add_argument('--output_wv', nargs='?', default='emb/karate.emb',
                        help='Embeddings wordvector path')
    parser.add_argument('--dimensions', type=int, default=128,
                        help='潜在特徴次元. Default is 200.')
    parser.add_argument('--walk_length', type=int, default=80,
                        help='１回のwalkで移動する回数. Default is 80.')
    parser.add_argument('--num_walks', type=int, default=10,
                        help='各ノードをソースとした時のwalkの試行回数. Default is 10.')
    parser.add_argument('--window_size', type=int, default=10,
                        help='Skip-gramのwindow幅. Default is 10.')
    parser.add_argument('--iter', default=500, type=int,
                        help='SGNSにおける目的関数最適化の最大epoch回数')
    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')
    parser.add_argument('--q', type=float, default=1,
                        help='Input hyperparameter. Default is 1.')
    parser.add_argument('--workers', type=int, default=3,
                        help='Number of parallel workers in learning the Skip-Gram model. Default is 3.')

    # bool値の場合，typeで指定すると誤った結果を引き起こす．ゆえにaction='store_true' or 'store_false'を用いる
    parser.add_argument('--is_weighted', action='store_true', help='Default is False.')
    parser.set_defaults(is_weighted=False)
    parser.add_argument('--is_directed', action='store_true', help='Default is False.')
    parser.set_defaults(is_directed=False)

    return parser.parse_args()


def read_graph():
    '''
    networkxを使って，グラフデータ(edgelist型)の読み込み
    weighted/directed の場合に，それぞれに合わせて読み込んでやる
    (注意) nodetypeは，intとする！（前処理でやる）[intとしてやらないとエラー出るかは要確認！！]
    '''
    if args.is_weighted:  # weighted graph の場合
        G = nx.read_edgelist(args.input, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
    else:  # unweighted graph の場合，重みを全て1に設定
        G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not args.is_directed:  # 基本的に，directedで定義して，ここでundirectedの場合に変換してやる
        G = G.to_undirected()

    return G


def learn_embeddings(walks, modelname):
    '''
    Skip-Gram model with Negative Sampling で学習
    '''
    walks = [list(map(str, walk)) for walk in walks]  # map(type, リスト)：リストの各要素をtypeに変換

    # inputdata: [[x1,x2,...], [y1,y2,...]] (系列データの集合で，各要素は str や intでよく異なる場合に異なるものと見なされる ⇨ 異なればそれだけの大きさの one-hotベクトル表現に)
    # sg: 1なら，skip-gram, 0なら，CBoW
    # min_count: n回未満登場する単語を破棄
    model = Word2Vec(walks, size=32, window=10, min_count=0, sg=1, negative=10, workers=3, iter=10)
    model.save('%s_model'%modelname)
    model.wv.save_word2vec_format('%s_wv'%modelname)

    return model



def main(datapath):
    '''
    コマンドラインから，入力のグラフデータを読み込んで，ランダムウォークで系列データを作成して，Skip-Gramで学習させる
    '''
    print('Reading edgelist...')
    nx_G = nx.read_edgelist(datapath, nodetype=str, create_using=nx.DiGraph())
    for edge in nx_G.edges():
        nx_G[edge[0]][edge[1]]['weight'] = 1

    print('Organizing baiased-weighted graph...')
    G = node2vec.Graph(nx_G, is_directed=True, p=1, q=1)  # node2vec.pyの中の，Graphクラスのインスタンスを生成, read_graphでunweightedでも"weight=1"と入れるため，args.is_unweightedは考えない．
    G.preprocess_transition_probs_deepwalk()  # 入力グラフのリンクの "重み・矢印 "に従って，各ノードごとにその重みを反映した遷移確率でウォーク出来るように準備

    print('Start random walk...')
    walks = G.simulate_deepwalks(20, 120)  # バイアス付きランダムウォークを開始

    print('Training the skip-gram model...')
    dataname = datapath.replace('.edgelist', '')
    model = learn_embeddings(walks, dataname)  # 上記で得られたノード系列データをインプットとして，skip-gramモデルで学習

    del walks, model
    gc.collect()


if __name__ == "__main__":

    datapath = 'data/yeast.edgelist'

    #datapaths = glob('data/*.edgelist')
    #datapaths[3:]
    #for i, datapath in enumerate(datapaths):
    start = time()

    main(datapath)

    print("ComputationTime：%.2f s"%(time()-start))
