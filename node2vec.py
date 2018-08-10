import numpy as np
import networkx as nx
import random

class Graph():
    """
    weighted graphを仮定．unweightedの場合は全て"weight=1"としたものをあらかじめ用意
    """
    def __init__(self, nx_G, is_directed, p, q):
        self.G = nx_G
        self.is_directed = is_directed
        self.p = p
        self.q = q


    def get_alias_edge(self, source, destination):
        """
        ・入力は，一つのエッジ (繋がっているノードの始点とノード終点)
        ・バイアス p, q を用いて，入力である1つの(向き付け)エッジに対して，終点ノードの近傍ノードそれぞれに移動するバイアス付き確率を計算し，それをalias_setup()の引数に入れたものを，返す
        ・重みは，正であることを仮定する
        """
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(destination)):  # ノード終点の近傍ノード全てを考える
            if dst_nbr == source:  # ソース(根元)に戻る確率を1/p倍 (ソースとの距離が0の場合)
                unnormalized_probs.append(G[destination][dst_nbr]['weight']/p)
            elif G.has_edge(dst_nbr, source):  # ソースとエッジを持つ ⇄ 距離が１ゆえ，バイアスなし
                unnormalized_probs.append(G[destination][dst_nbr]['weight'])
            else:  # それ以外は，ソースとの距離が2の場合，1/q倍
                unnormalized_probs.append(G[destination][dst_nbr]['weight']/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]  # 入力である1つの(向き付け)エッジに対して，終点ノードの近傍ノードそれぞれに移動するバイアス付き確率が格納（ソート順で）

        return alias_setup(normalized_probs)  # バイアス付きの重みエッジに対応した(J,q)（ ⇨ これをalias_draw()に入れてやると1回抽出，複数回入れることで復元抽出になる）



    def preprocess_transition_probs(self):
        """
        グラフのリンクの重みに従って，各ノードごとにその重みを反映した遷移確率でウォーク出来るように準備
        具体的には，
        ・各ノードごとに，そのneighborsへ重みに基づいた確率に従って復元抽出できるように，alias_nodesに（J，q）を格納，これはランダムウォークの出発点でだけ使用する（なぜなら，出発点ではバイアスp,qは関係しないので）
        ・各エッジごとに，その終点ノードのneighborsのバイアス付き重みに従って復元抽出できるように，alias_edgesに（J，q）を格納，ランダムウォークの出発点以外ではこっちを使って復元抽出を行う
        """
        G = self.G
        is_directed = self.is_directed

        # alias_nodes: 各ノードごとの近傍ノードに対応する(J,q)を要素に持つ
        # (これをalias_draw(J,q)に入れれば，重みに基づいた復元抽出になる)
        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))] # ノードのneighborが持つ重みをリストに．注意として，directedの場合考えているノードがエッジの始点であるようなもののみneighborsとちゃんと見なされるので大丈夫
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        # alias_edges: バイアス p,q の重み付けしたエッジに基づく(J,q)
        alias_edges = {}
        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])  # undirectedの場合，始点と終点は区別ないので，edgeの始点[0番目の要素]と終点[1番目の要素]をひっくり返したやつで定義

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return



    def node2vec_walk(self, walk_length, start_node):
        """
        入力：1回のウォークの長さ，出発ノード
        出力：バイアス付き重みありランダムウォークから得られる系列
        出発ノードから，ランダムウォークを走らせ，ノード列を得る
        """
        G = self.G
        alias_nodes = self.alias_nodes  # 各ノードごとに，そのneighborsへ重みに基づいた確率に従って復元抽出できるように，alias_nodesに（J，q）を格納，これはランダムウォークの出発点でだけ使用する（なぜなら，出発点ではバイアスp,qは関係しないので）
        alias_edges = self.alias_edges  # 各エッジごとに，その終点ノードのneighborsのバイアス付き重みに従って復元抽出できるように，alias_edgesに（J，q）を格納，ランダムウォークの出発点以外ではこっちを使って復元抽出を行う

        walk = [start_node]  # ランダムウォークで得られるノード列の初期化

        # 1つずつランダムウォークを行っていく
        while len(walk) < walk_length:
            cur = walk[-1]  # 現在地のノード
            cur_nbrs = sorted(G.neighbors(cur))  # 現在のノードに接しているノードリスト

            if len(cur_nbrs) > 0:
                if len(walk) == 1:  # 一番初めだけは，バイアス付かないので特別
                    # バイアスなしでランダム抽出されたNNを加える
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    # nextは，バイアス付きで選ばれたneighborが加わる
                    prev = walk[-2]
                    next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0],
                                                alias_edges[(prev, cur)][1])]
                    walk.append(next)
            else:
                break

        return walk



    def simulate_walks(self, num_walks, walk_length):
        '''
        入力：ウォークを行う回数(全てのノードを始点として一通り実行して１回とカウント)，1回のウォークの長さ
        出力：系列（ノード列）の集合
        node2vec_walk()というメソッドを繰り返し行い，系列の集合を得る
        '''
        G = self.G
        walks = []
        nodes = list(G.nodes())
        print("Walk iteration:")
        for walk_iter in range(num_walks):
            print(str(walk_iter+1), '/', str(num_walks))
            random.shuffle(nodes)  # ?
            for node in nodes:  # 全てのノードを始点にして，ウォークをそれぞれ求める（これで１回とカウント）
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

        return walks


########################################################################################################

    def preprocess_transition_probs_deepwalk(self):
        """
        グラフのリンクの重みに従って，各ノードごとにその重みを反映した遷移確率でウォーク出来るように準備
        具体的には，
        ・各ノードごとに，そのneighborsへ重みに基づいた確率に従って復元抽出できるように，alias_nodesに（J，q）を格納，これはランダムウォークの出発点でだけ使用する（なぜなら，出発点ではバイアスp,qは関係しないので）
        ・各エッジごとに，その終点ノードのneighborsのバイアス付き重みに従って復元抽出できるように，alias_edgesに（J，q）を格納，ランダムウォークの出発点以外ではこっちを使って復元抽出を行う
        """
        G = self.G
        is_directed = self.is_directed

        # alias_nodes: 各ノードごとの近傍ノードに対応する(J,q)を要素に持つ
        # (これをalias_draw(J,q)に入れれば，重みに基づいた復元抽出になる)
        alias_nodes = {}
        L = len(G.nodes())
        for i, node in enumerate(G.nodes()):
            print('%d/%d'%(i+1, L))
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))] # ノードのneighborが持つ重みをリストに．注意として，directedの場合考えているノードがエッジの始点であるようなもののみneighborsとちゃんと見なされるので大丈夫
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        # alias_edges: バイアス p,q の重み付けしたエッジに基づく(J,q)
        alias_edges = {}
        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])  # undirectedの場合，始点と終点は区別ないので，edgeの始点[0番目の要素]と終点[1番目の要素]をひっくり返したやつで定義

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return

    def deep_walk(self, walk_length, start_node):
        """
        入力：1回のウォークの長さ，出発ノード
        出力：バイアス付き重みありランダムウォークから得られる系列
        出発ノードから，ランダムウォークを走らせ，ノード列を得る
        """
        G = self.G
        alias_nodes = self.alias_nodes  # 各ノードごとに，そのneighborsへ重みに基づいた確率に従って復元抽出できるように，alias_nodesに（J，q）を格納，これはランダムウォークの出発点でだけ使用する（なぜなら，出発点ではバイアスp,qは関係しないので）
        #alias_edges = self.alias_edges  # 各エッジごとに，その終点ノードのneighborsのバイアス付き重みに従って復元抽出できるように，alias_edgesに（J，q）を格納，ランダムウォークの出発点以外ではこっちを使って復元抽出を行う

        walk = [start_node]  # ランダムウォークで得られるノード列の初期化

        # 1つずつランダムウォークを行っていく
        while len(walk) < walk_length:
            cur = walk[-1]  # 現在地のノード
            cur_nbrs = sorted(G.neighbors(cur))  # 現在のノードに接しているノードリスト

            if len(cur_nbrs) > 0:
                # バイアスなしでランダム抽出されたNNを加える
                walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])

            else:
                break

        return walk


    def simulate_deepwalks(self, num_walks, walk_length):
        '''
        入力：ウォークを行う回数(全てのノードを始点として一通り実行して１回とカウント)，1回のウォークの長さ
        出力：系列（ノード列）の集合
        node2vec_walk()というメソッドを繰り返し行い，系列の集合を得る
        '''
        G = self.G
        walks = []
        nodes = list(G.nodes())
        print("Walk iteration:")
        for walk_iter in range(num_walks):
            print(str(walk_iter+1), '/', str(num_walks))
            random.shuffle(nodes)  # ?
            for node in nodes:  # 全てのノードを始点にして，ウォークをそれぞれ求める（これで１回とカウント）
                walks.append(self.deep_walk(walk_length=walk_length, start_node=node))

        return walks



### Walker's Alias Method (復元抽出の高速アルゴリズム) ###

# うまく重み付き変数を分割して並べ替えてやって，"2つの変数を含むブロック"で等分割されるようにする
# すれば，O(1)で復元抽出が可能になる（通常，2分探索でO(logk)）


def alias_setup(probs):
    """
    確率の重みのリスト(足して１)を渡すと，
    J:どのブロックにどれで補ったかの情報が書かれたリスト，q:各ブロックの前半部に含まれる確率を，返す
    """
    n = len(probs)
    q = np.zeros(n)  # 各要素の確率をリストの長さ倍したものを格納するもの
    J = np.zeros(n, dtype=np.int)

    smaller = []
    larger = []
    for i, prob in enumerate(probs):
        q[i] = n * prob
        if q[i] < 1.0:
            smaller.append(i)
        else:
            larger.append(i)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop(-1)  # リストsmallerの末尾の要素を削除し，smallには削除された末尾が入る
        large = larger.pop(-1)

        # 1より大きいものと，1より小さいものを一対一のペアで割り振りを考えていく：
        J[small] = large  # どこにどれを割り振ったかを記録
        q[large] = q[large] - (1.0 - q[small])  # 1よりデカかったものを，小さいやつに振り分けてく，その余りを入れる

        # 割り振った後の，大きいやつの余りが1より大きいかどうか：
        if q[large] < 1.0:
            smaller.append(large)  # 1より小さくなったら，もう振り分けないので，新たなブロックとして加える
        else:
            larger.append(large)  # 1より大きかったら，また1より大きいので，割り振るリストとして加える

    return J, q



def alias_draw(J, q):
    """
    alias_setup(probs)で作った J,qを使って，１回抽出（ ⇨ これを繰り返せば復元抽出）
    """
    n  = len(J)

    k = int(np.floor(np.random.rand() * n))  # 一様に「0 ~ n-1」のどれかの整数を返す (np.floorは切捨て)

    if np.random.rand() < q[k]:  # q[k]は1より小さい確率で必ずブロックの始めに入るので，それより小さかったら，k
        return k
    else:  # 大きかったら，k番目の中の補ったやつの番号，J[k]
        return J[k]
