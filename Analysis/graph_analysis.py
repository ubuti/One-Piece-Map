import re
import emoatlas
import networkx as nx
import numpy as np
import emoatlas
from emoatlas.valence import english as english_valence


class Graph_Analysis:
    """Helper class to produce network properties of a formamentis network for english language. Relies on the networkx library. Uses "en_core_web_lg" as parsing model. Look out for naming conflicts as I use "round"."""
    
    # Class object
    emos = emoatlas.EmoScores(language='english', spacy_model='en_core_web_lg')

    def __init__(self, text, distance, **kwargs):
        self.fmn = Graph_Analysis.emos.formamentis_network(
            text=text, **kwargs, max_distance=distance)
        self.distance = distance
        self.text = text
        self.nodes = self.fmn.vertices
        self.edges = self.fmn.edges
        self.graph = nx.Graph(self.edges)
        # Network attributes
        self.ratio_largest_connected_component = None
        self.number_components = None
        self.mean_deg = None
        self.min_deg = None
        self.max_deg = None
        self.mean_degree_assortativity = None
        self.mean_cluster_coefficient = None
        self.algrebaric_connectivity = None
        self.mean_closeness_centrality = None
        self.weighted_closeness_centrality = None
        self.max_closeness_centrality = None
        self.number_highly_between_nodes = None
        self.size_largest_component = None
        self.pos_words_count = None
        self.neg_words_count = None
        self.neutral_words_count = None
        self.pos_words_ratio = None
        self.neg_words_ratio = None
        self.pos_words_assortativity = None
        self.neg_words_assortativity = None
        self.neu_words_assortativity = None
        self.pos_words = english_valence._positive
        self.neg_words = english_valence._negative
        self.neutral_words = english_valence._ambivalent
        self.number_of_nodes = None
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
        self.max_clique_size = None
        self.local_efficiency = None
        self.global_efficiency = None
        self.is_chordal = None
        self.min_edge_cover = None
        self.transitivity = None
        self.max_independent_set = None
        self.number_triangles = None
        self.triangles_pos_ratio = 0
        self.triangles_neg_ratio = 0
        self.triangles_neutral_ratio = 0
        self.modularity = None
        self.core_size_ratio = None
        self.core_size_absolute = None
        # Call setting
        self.calculate_attributes()

    def plot_graph(self, layout='force_layout', **kwargs):
        """Wraper function of draw_formamentis. Parameter:
        fmn=formamentis network
        layout='force_layout' (default) or 'circular_layout',
        highlight=[], list of network nodes
        thickness=1, int
        ax=None, matplotlib axis
        translated=False, bool
        alpha_syntactic=0.5, float
        alpha_hypernyms=0.5, float
        alpha_synonyms=0.5. float
        """
        Graph_Analysis.emos.draw_formamentis(self.fmn, layout=layout, **kwargs)

    def set_emotions(self):
        """Uses the zscores for emotion from emoatlas. Sets the zscores for each emotion as an instance attribute."""
        self.anger = Graph_Analysis.emos.zscores(self.fmn)['anger']
        self.trust = Graph_Analysis.emos.zscores(self.fmn)['trust']
        self.disgust = Graph_Analysis.emos.zscores(self.fmn)['disgust']
        self.surprise = Graph_Analysis.emos.zscores(self.fmn)['surprise']
        self.joy = Graph_Analysis.emos.zscores(self.fmn)['joy']
        self.anticipation = Graph_Analysis.emos.zscores(self.fmn)['anticipation']
        self.fear = Graph_Analysis.emos.zscores(self.fmn)['fear']
        self.sadness = Graph_Analysis.emos.zscores(self.fmn)['sadness']

    def set_density(self):
        """Sets the density of the network. Represents the proportion of existing connections to the total number of possible connections. Range between 0 and 1."""
        self.density = nx.density(self.graph)
        
    def set_centrality_importance(self):
        """Sets the importance of the centrality of the nodes in the network. The centrality importance is the sum of the betweenness centrality of the nodes multiplied by the degree of the nodes. The degree centrality importance is the sum of the degree of the nodes. The pos_centrality_importance is the sum of the centrality importance of the positive nodes according to pos_words dictionary. The neg_centrality_importance is the sum of the centrality importance of the negative nodes. Division by the number of nodes is necessary to get the average centrality importance."""
        betweenness_nodes = nx.betweenness_centrality(self.graph, k=5)
        total_sum_degree = 0.0
        for key, value in betweenness_nodes.items():
            total_sum_degree += value * nx.degree(self.graph, nbunch=key)
            if key in self.pos_words:
                self.pos_centrality_importance += value * nx.degree(self.graph, nbunch=key)
            elif key in self.neg_words:
                self.neg_centrality_importance += value * nx.degree(self.graph, nbunch=key)
                    
        self.degree_centrality_importance = total_sum_degree / len(self.nodes)
        self.pos_centrality_importance /= len(self.nodes) # consider omitting division
        self.neg_centrality_importance /= len(self.nodes) # consider omitting division
    
    def set_number_of_nodes(self):
        """Self explanatory."""
        self.number_of_nodes = len(self.graph.nodes)
        
    def set_average_path_length(self):
        """Sets the average path length of the network. Represents the average number of steps along the shortest paths for all possible pairs of network nodes. Weighted by the component size to factor in its importance + normalization by divison of total number of components at the end."""
        total_sum = 0.0
        for component in nx.connected_components(self.graph):
            total_sum += nx.average_shortest_path_length(self.graph.subgraph(component)) * len(component)
        self.average_path_length = total_sum #/ len([gen for gen in nx.connected_components(self.graph)])

    def set_valence_count(self):
        """Sets the number of positive and negative nodes in the network. Some nodes are in no lexicon, thus they are considered neutral."""
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
        """Sets the ratio of positive and negative nodes to the total number of nodes in the network."""
        self.pos_words_ratio = self.pos_words_count/len(self.nodes)
        self.neg_words_ratio = self.neg_words_count/len(self.nodes)

    def set_valence_to_nodes(self):
        """Sets valence attributes to each node. """
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
        """Sets the ratio of edges between equal valences. To be changed to represent true assortativty [-1, 1] in the future."""
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
      """Sets the ratio of the largest connected component of the network to the total number of nodes. This value represents the proportion of nodes that can be reached from an arbitrary node. Range between 0 and 1."""
      self.ratio_largest_connected_component = max(len(c) for c in nx.connected_components(
          self.graph)) / len(self.nodes)

    def set_number_components(self):
        """Sets the number of components in the network."""
        self.number_components = nx.number_connected_components(self.graph)

    def set_mean_degree(self):
        """Sets the average degree of the network."""
        degrees = [degree for node, degree in self.graph.degree()]
        self.mean_deg = round(np.mean(degrees), 2)

    def set_max_degree(self):
        """Sets the maximum degree of the network."""
        degrees = [degree for node, degree in self.graph.degree()]
        if len(degrees) == 0:
            self.max_deg = 0
        else:
            self.max_deg = int(np.max(degrees))

    def set_min_degree(self):
        """Sets the minimum degree of the network."""
        degrees = [degree for node, degree in self.graph.degree()]
        if len(degrees) == 0:
            self.min_deg = 0
        else:
            self.min_deg = float(np.min(degrees))

    def set_mean_degree_assortativity(self):
        """Sets the average degree assortativity of the network."""
        degrees = [degree for node, degree in self.graph.degree()]
        self.mean_degree_assortativity = nx.degree_pearson_correlation_coefficient(self.graph, x=degrees, y=degrees)

    def set_mean_cluster_coef(self):
        """Sets the average clustering coefficient of the network. Represents the tendency of nodes to cluster together.
       
       # prown to error
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
        self.mean_cluster_coefficient = round(
            Ci / len(self.nodes), 2) if len(self.nodes) > 0 else 0
        # end 
            """
        self.mean_cluster_coefficient = nx.average_clustering(self.graph)
        

    def set_algebraic_connectivity(self):
        """The algebraic connectivity of the network. Represents the connectivity of the network in terms of robustness to edge removal. Incorporates the size of the removed component."""
        try:
            self.algrebaric_connectivity = nx.algebraic_connectivity(self.graph, weight=None)
        except Exception as e:
            print(f'Can not make self.alg_con because of {e}.')
            self.algrebaric_connectivity = 0.0

    def set_mean_closeness_centrality(self):
        """Sets the average closeness centrality of the network. Represents the average distance of a node to all other nodes."""
        self.mean_closeness_centrality = round(np.mean(list(nx.closeness_centrality(self.graph).values(
        ))), 2) if len(list(nx.closeness_centrality(self.graph).values())) > 0 else 0.0

    def set_weighted_betweenness_centrality(self):
        """Sets average betweenness centrality of the network. Represents the average number of shortest paths that pass through a node."""
        components = nx.connected_components(self.graph)
        between_centrality = 0.0
        for component in components:
            subgraph = self.graph.subgraph(component)
            mean = np.mean(list(nx.betweenness_centrality(subgraph).values()))
            between_centrality += mean * len(subgraph)
        self.weighted_closeness_centrality = round(between_centrality, 2)

    def set_max_closeness_centrality(self):
        """Sets the maximum value among the closeness centrality and thus the most central node."""
        self.max_closeness_centrality = round(np.max(list(nx.closeness_centrality(self.graph).values(
        ))), 2) if len(list(nx.closeness_centrality(self.graph).values())) > 0 else 0.0

    def set_highly_between_nodes(self):
        """Returns the sum of highly between nodes. Represents the sum of nodes with a betweenness centrality of 0.5 or higher. 0 if not present."""
        components = nx.connected_components(self.graph)
        high_betweenness = 0.0
        for component in components:
            subgraph = self.graph.subgraph(component)
            filtered_values = [value for value in nx.betweenness_centrality(
                subgraph).values() if value >= 0.5]
            high_betweenness += np.sum(filtered_values)
        self.number_highly_between_nodes = high_betweenness

    def set_size_largcomp(self):
        """Size of the largest connected component of the network."""
        self.size_largest_component = max(len(c) for c in nx.connected_components(self.graph))

    ####################################################################################
    # New measures to be added
    
    def set_nodes_to_age_ratio(self):
        #self.nodes_to_age_ratio = len(self.nodes) / self.age
        pass
    
    def set_maximum_clique_size(self):
        G = nx.find_cliques(self.graph)
        self.max_clique_size = max([len(gn) for gn in G])

    def set_local_efficiency(self):
        self.local_efficiency = nx.local_efficiency(self.graph)
        
    def set_global_efficiency(self):
        self.global_efficiency = nx.global_efficiency(self.graph)
        
    def set_is_chordal(self):
        self.is_chordal = nx.is_chordal(self.graph)
        
    def set_min_edge_cover(self):
        self.min_edge_cover = len(nx.min_edge_cover(self.graph))
        
    def set_transitivity(self):
        self.transitivity = nx.transitivity(self.graph)
        
    def set_maximal_independent_set(self):
        self.max_independent_set = len(nx.maximal_independent_set(self.graph))
        
    def set_total_number_of_triangles(self):
        self.number_triangles = sum(nx.triangles(self.graph).values())
        
    def set_triangle_valence_ratio(self):
        for key, value in nx.triangles(self.graph).items():
            if key in self.pos_words:
                self.triangles_pos_ratio += value
            elif key in self.neg_words:
                self.triangles_neg_ratio += value
            elif key in self.neutral_words:
                self.triangles_neutral_ratio += value
            else:
                self.triangles_neutral_ratio += value
        
        self.triangles_pos_ratio /= self.number_triangles
        self.triangles_neg_ratio /= self.number_triangles
        self.triangles_neutral_ratio /= self.number_triangles
        
                
    def set_modularity(self):
        communities = nx.community.louvain_communities(self.graph)
        self.modularity = nx.community.modularity(self.graph, communities)
        
    def set_core_size(self):
        self.core_size_absolute = len(nx.k_core(self.graph, self.distance))
        self.core_size_ratio = self.core_size_absolute / len(self.nodes)
        
    def is_kl_connected(self):
        pass
    
    ####################################################################################

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
      self.set_highly_between_nodes()
      self.set_size_largcomp()
      self.set_valence_count()
      self.set_valence_ratio()
      self.pos_words_assortativity = self.set_valence_assortativity(sentiment='pos')
      self.neg_words_assortativity = self.set_valence_assortativity(sentiment='neg')
      self.neu_words_assortativity = self.set_valence_assortativity(
          sentiment='neu')
      self.set_density()
      self.set_centrality_importance()
      self.set_number_of_nodes()
      self.set_average_path_length()
      self.set_emotions()
      # bring in 
      self.set_maximum_clique_size()
      self.set_local_efficiency()
      self.set_global_efficiency()
      self.set_is_chordal()
      self.set_min_edge_cover()
      self.set_transitivity()
      self.set_maximal_independent_set()
      self.set_total_number_of_triangles()
      self.set_triangle_valence_ratio()
      self.set_modularity()
      self.set_core_size()

    ####################################################################################


class Emo_Lexicon:
    """Helper class to create a valence dictionary from a text file. Used to process EmoLex or the NRC Emotion Lexicon."""
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
