import re
import emoatlas
import networkx as nx
import numpy as np
import emoatlas
from emoatlas.valence import english as english_valence


class Graph_Analysis:
    """Helper class to calculate network properties of a formamentis network. It heavily relies on the networkx library. For now, this is okay. But the goal is to replace these functions with own implementations. 
    Look out for naming conflicts as I use 'round'. Feel free to contribute by adding, verifying and improving the attribute calculating-functions which are the heart of this class."""
    emos = emoatlas.EmoScores(language='english', spacy_model='en_core_web_lg')

    def __init__(self, text, distance, **kwargs):
        # Class object
        self.fmn = Graph_Analysis.emos.formamentis_network(
            text=text, **kwargs, max_distance=distance)
        self.text = text
        self.nodes = self.fmn.vertices
        self.edges = self.fmn.edges
        self.graph = nx.Graph(self.edges)
        # self.lexicon = lexicon needed for the emotional valence
        # Attributes for prediction
        self.lcc_ratio = None
        self.num_comp = None
        self.mean_deg = None
        self.min_deg = None
        self.max_deg = None
        self.mean_deg_ast = None
        self.mean_clust_coef = None
        self.alg_con = None
        self.mean_close_cent = None
        self.weight_close_cent = None
        self.max_close_cent = None
        self.num_bet_nodes = None
        self.slc = None
        self.pos_words_count = None
        self.neg_words_count = None
        self.neutral_words_count = None
        self.pos_words_ratio = None
        self.neg_words_ratio = None
        self.pos_words_ast = None
        self.neg_words_ast = None
        self.neu_words_ast = None
        self.pos_words = english_valence._positive
        self.neg_words = english_valence._negative
        self.neutral_words = english_valence._ambivalent
        # new ones
        self.num_nodes = None
        self.density = None
        self.average_path_length = None
        self.degree_centrality_importance = None # bewteenness * degree 
        self.pos_centrality_importance = 0.0
        self.neg_centrality_importance = 0.0
        # emotions 
        self.anger = None
        self.trust = None
        self.disgust = None
        self.joy = None
        self.fear = None
        self.sadness = None
        self.anticipation = None
        self.surprise = None
        # Call setting
        self.calculate_attributes()

    def plot_graph(self, layout='force_layout', **kwargs):
        """Nice plotting function. Choose between 'force_layout' and 'circular_layout' for the layout. Parameters are:
        fmn, layout='edge_bundling', highlight=[], thickness=1, ax=None, translated=False, alpha_syntactic=0.5, alpha_hypernyms=0.5, alpha_synonyms=0.5."""
        Graph_Analysis.emos.draw_formamentis(self.fmn, layout=layout, **kwargs)

    def set_emotions(self):
        self.anger = Graph_Analysis.emos.zscores(self.fmn)['anger']
        self.trust = Graph_Analysis.emos.zscores(self.fmn)['trust']
        self.disgust = Graph_Analysis.emos.zscores(self.fmn)['disgust']
        self.surprise = Graph_Analysis.emos.zscores(self.fmn)['surprise']
        self.joy = Graph_Analysis.emos.zscores(self.fmn)['joy']
        self.anticipation = Graph_Analysis.emos.zscores(self.fmn)['anticipation']
        self.fear = Graph_Analysis.emos.zscores(self.fmn)['fear']
        self.sadness = Graph_Analysis.emos.zscores(self.fmn)['sadness']

    def set_density(self):
        self.density = nx.density(self.graph)
        
    def set_spread_activation_centrality(self):
        betweenness_nodes = nx.betweenness_centrality(self.graph, k=5)
        total_sum_degree = 0.0
        for key, value in betweenness_nodes.items():
            total_sum_degree += value * nx.degree(self.graph, nbunch=key)
            if key in self.pos_words:
                self.pos_centrality_importance += value * nx.degree(self.graph, nbunch=key)
            elif key in self.neg_words:
                self.neg_centrality_importance += value * nx.degree(self.graph, nbunch=key)
                    
        self.degree_centrality_importance = total_sum_degree / len(self.nodes)
        self.pos_centrality_importance /= len(self.nodes)
        self.neg_centrality_importance = len(self.nodes)
    
    def set_num_nodes(self):
        self.num_nodes = len(self.graph.nodes)
        
    def set_average_path_length(self):
        total_sum = 0.0
        for component in nx.connected_components(self.graph):
            total_sum += nx.average_shortest_path_length(self.graph.subgraph(component)) * len(component)
        self.average_path_length = total_sum / len(self.nodes) # or divide by len(nx.connected_components(self.graph))

    def set_valence_count(self):
        """Retunrs the number of psotive and negative nodes in the network. Some nodes are in no lexicon, so they are considered neutral."""
        pos_counter = 0
        neg_counter = 0
        for node in self.nodes:
            try:
                if node in self.pos_words:
                    pos_counter += 1
                if node in self.neg_words:
                    neg_counter += 1
            except:
                pass
        self.neutral_words_count = len(self.nodes) - pos_counter - neg_counter
        self.pos_words_count = pos_counter
        self.neg_words_count = neg_counter

    def set_valence_ratio(self):
        """Returns the ratio of positive and negative nodes to the total number of nodes in the network."""
        self.pos_words_ratio = self.pos_words_count/len(self.nodes)
        self.neg_words_ratio = self.neg_words_count/len(self.nodes)

    def set_valence_to_nodes(self):
        """Takes a formamentis network and sets valence attributes to the nodes. """
        for node in self.graph:
            attr = "pos" if node in self.pos_words else "neg" if node in self.neg_words else "neu"
            nx.set_node_attributes(self.graph, {node: attr}, 'valence')
        neu_nodes = [node for node, attr in self.graph.nodes(
            data=True) if attr['valence'] == 'neu']
        pos_nodes = [node for node, attr in self.graph.nodes(
            data=True) if attr['valence'] == 'pos']
        neg_nodes = [node for node, attr in self.graph.nodes(
            data=True) if attr['valence'] == 'neg']
        # Make sure node is at most in one set
        intersec_nodes = list(set(neu_nodes) & set(pos_nodes) & set(neg_nodes))
        assert len(intersec_nodes) == 0

    def set_valence_assortativity(self, sentiment):
        """Returns the ratio of edges between equal valences."""
        self.set_valence_to_nodes()
        if sentiment == 'all':
            same_valence_edges = 0.0
            for edge in self.graph.edges:
                if self.graph.nodes[edge[0]]['valence'] == self.graph.nodes[edge[1]]['valence']:
                    same_valence_edges += 1
            valence_assortativity = same_valence_edges / len(self.graph.edges)
            return valence_assortativity
        if sum([attr['valence'] == sentiment for _, attr in self.graph.nodes(data=True)]) == 0:
            return 0
        same_valence_edges = 0.0
        num_edges = 0.0
        for edge in self.graph.edges:
            if self.graph.nodes[edge[0]]['valence'] == sentiment or self.graph.nodes[edge[1]]['valence'] == sentiment:
                num_edges += 1
                if self.graph.nodes[edge[0]]['valence'] == self.graph.nodes[edge[1]]['valence']:
                    same_valence_edges += 1
        valence_assortativity = same_valence_edges / num_edges
        return valence_assortativity

    def set_ratio_largest_connected_component(self) :
      """Calculate the ratio of the largest connected component of the network to the total number of nodes. This value represents the proportion of nodes that can be reached from an arbitrary node. Range between 0 and 1."""
      self.lcc_ratio = max(len(c) for c in nx.connected_components(
          self.graph)) / len(self.nodes)

    def set_number_components(self):
        """Number of components in the network. Represents the number of subgraphs in the network."""
        self.num_comp = nx.number_connected_components(self.graph)

    def set_mean_degree(self):
        """Calculate the average degree of the network. This value represents the average number of connections of a node."""
        degrees = [degree for node, degree in self.graph.degree()]
        self.mean_deg = round(np.mean(degrees), 2)

    def set_max_degree(self):
        """Calculate the maximum degree of the network. This value represents the maximum number of connections of a node."""
        degrees = [degree for node, degree in self.graph.degree()]
        if len(degrees) == 0:
            self.max_deg = 0
        else:
            self.max_deg = int(np.max(degrees))

    def set_min_degree(self):
        """Calculate the minimum degree of the network. This value represents the minimum number of connections of a node."""
        degrees = [degree for node, degree in self.graph.degree()]
        if len(degrees) == 0:
            self.min_deg = 0
        else:
            self.min_deg = float(np.min(degrees))

    def set_mean_degree_assortativity(self):
        """The average degree assortativity of the network."""
        degrees = [degree for node, degree in self.graph.degree()]
        self.mean_deg_ast = nx.degree_pearson_correlation_coefficient(self.graph, x=degrees, y=degrees)

    def set_mean_cluster_coef(self):
        """The average local clustering coefficient of the network. Represents the tendency of nodes to cluster together."""
        Ci = 0.0
        for node in self.graph.nodes:
            Ki = list(nx.neighbors(self.graph, node))
            Ni = 0.0
            for edge in self.graph.edges:
                if edge[0] in Ki and edge[1] in Ki:
                    Ni += 1
            try:
                Ci += 2 * Ni / (len(Ki) * (len(Ki) - 1))
            except ZeroDivisionError:
                # print(f'Can not divide by {len(Ki) * len(Ki)-1}. Node {node} has no neighbors: {Ki}.')
                continue
        self.mean_clust_coef = round(
            Ci / len(self.nodes), 2) if len(self.nodes) > 0 else 0

    def set_algebraic_connectivity(self):
        """The algebraic connectivity of the network. Represents the connectivity of the network in terms of robustness to edge removal. Incorporates the size of the removed component."""
        try:
            self.alg_con = nx.algebraic_connectivity(self.graph, weight=None)
        except Exception as e:
            print(f'Can not make self.alg_con because of {e}.')
            self.alg_con = 0.0

    def set_mean_closeness_centrality(self):
        """Average closeness centrality of the network. Represents the average distance of a node to all other nodes."""
        self.mean_close_cent = round(np.mean(list(nx.closeness_centrality(self.graph).values(
        ))), 2) if len(list(nx.closeness_centrality(self.graph).values())) > 0 else 0.0

    def set_weighted_betweenness_centrality(self):
        """Average betweenness centrality of the network. Represents the average number of shortest paths that pass through a node."""
        components = nx.connected_components(self.graph)
        between_centrality = 0.0
        for component in components:
            subgraph = self.graph.subgraph(component)
            mean = np.mean(list(nx.betweenness_centrality(subgraph).values()))
            between_centrality += mean * len(subgraph)
        self.weight_close_cent = round(between_centrality, 2)

    def set_max_closeness_centrality(self):
        """Return the maximum value from a list of numbers."""
        self.max_close_cent = round(np.max(list(nx.closeness_centrality(self.graph).values(
        ))), 2) if len(list(nx.closeness_centrality(self.graph).values())) > 0 else 0.0

    def num_between_nodes(self):
        """Returns the sum of highly between nodes. Represents the sum of nodes with a betweenness centrality of 0.75 or higher. 0 if not present."""
        components = nx.connected_components(self.graph)
        high_betweenness = 0.0
        for component in components:
            subgraph = self.graph.subgraph(component)
            filtered_values = [value for value in nx.betweenness_centrality(
                subgraph).values() if value >= 0.75]
            high_betweenness += np.sum(filtered_values)
        self.num_bet_nodes = high_betweenness

    def set_size_largcomp(self):
        """Size of the largest connected component of the network."""
        self.slc = max(len(c) for c in nx.connected_components(self.graph))

    def calculate_attributes(self):
      self.set_ratio_largest_connected_component()
      self.set_number_components()
      self.set_max_degree()
      self.set_min_degree()
      self.set_mean_degree_assortativity()
      self.set_mean_cluster_coef()
      self.set_mean_degree()
      self.set_algebraic_connectivity()
      self.set_mean_closeness_centrality()
      self.set_weighted_betweenness_centrality()
      self.set_max_closeness_centrality()
      self.num_between_nodes()
      self.set_size_largcomp()
      self.set_valence_count()
      self.set_valence_ratio()
      self.pos_words_ast = self.set_valence_assortativity(sentiment='pos')
      self.neg_words_ast = self.set_valence_assortativity(sentiment='neg')
      self.neu_words_ast = self.set_valence_assortativity(sentiment='neu')
      self.set_density()
      self.set_spread_activation_centrality()
      self.set_num_nodes()
      self.set_average_path_length()
      self.set_emotions()

    ####################################################################################


class Emo_Lexicon:
    """Helper class to create a valence dictionary from a text file. Since there is now valence in Graph_Analysis it is not really needed any longer. Delete it when I am certain of that"""
    
    def __init__(self, path, num_val=True):
        """Here comes the documentation"""
        
        with open(str(path), 'r') as file:
            content = file.read()

        filtered_content = '\n'.join(line for line in content.split(
            '\n') if 'positive' in line or 'negative' in line)

        positive_dict = {}
        negative_dict = {}

        pattern = r'^\S+'

        for line in filtered_content.split('\n'):
            if 'positive' in line:
                match = re.match(pattern, line)
                key = match.group()
                value = int(line[-1])
                positive_dict[key] = value
            else:
                match = re.match(pattern, line)
                key = match.group()
                value = int(line[-1])
                if value == 1:
                    negative_dict[key] = -1
                else:
                    negative_dict[key] = value

        valence_dict = {}

        for word in list(positive_dict.keys()) + list(negative_dict.keys()):
            try:
                valence_dict[word] = positive_dict[word] + negative_dict[word]
            except:
                valence_dict.update(
                    {word: int(positive_dict[word] + negative_dict[word])})

            if num_val == False:
                if valence_dict[word] == 0:
                    valence_dict[word] = 'neutral'
                elif valence_dict[word] == 1:
                    valence_dict[word] = 'positive'
                elif valence_dict[word] == -1:
                    valence_dict[word] = 'negative'
                else:
                    print('Error for {vertex}')
                    break

        self.dict = valence_dict
