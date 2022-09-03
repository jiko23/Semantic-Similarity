from collections import defaultdict
from importlib.resources import path
from turtle import distance

INF = int(0x3f3f3f3f)
class KnowledgeGraph:
    def __init__(self, text_list, reason_list):
        self.__knowledge_graph = defaultdict(list)
        self.__text_list = text_list
        self.__reason_list = reason_list
    
    def __getGraph(self, parse_tree_group):

        for tree in parse_tree_group:
            for node in tree:
                for adjacent_node in tree:
                    if adjacent_node[0] != node[0]:
                        #self.__knowledge_graph[node[0]].append((adjacent_node[0], 1))
                        self.__knowledge_graph[node[0]].append(adjacent_node[0])
                    #else:
                        #self.__knowledge_graph[node[0]].append((adjacent_node[0], 0))
        return self.__knowledge_graph

    def buildGraph(self):
        total_adjacency_list = self.__text_list + self.__reason_list
        graph  = self.__getGraph(total_adjacency_list)
        return graph
    
        
    def __getShortestPath(self, start, goal):
        explored = []

        queue = [[start]]
        
        if start == goal:
           # print("Same Node")
            return [goal]
        
        while queue:
            path = queue.pop(0)
            node = path[-1]
            
            if node not in explored:
                neighbours = self.__knowledge_graph[node]
                

                for neighbour in neighbours:
                    new_path = list(path)
                    new_path.append(neighbour)
                    queue.append(new_path)
                    

                    if neighbour == goal:
                        #print("Shortest path = ", new_path)
                        return new_path
                explored.append(node)
    
        return list()
    
    def getSemanticSimilary(self, reason, text):
        similarity_nodes = []
        for source_node in reason:
            for dest_node in text:
                if(source_node[1] == dest_node[1]):
                    tmp = self.__getShortestPath(source_node[0], dest_node[0])
                # allowing maximum of two nodes distance between source and destination nodes.
                    if(len(tmp) != 0 and len(tmp) < 4):
                        if source_node[0] not in similarity_nodes:
                            similarity_nodes.append(source_node[0])
        #print("reason {}, text {}".format(reason, text))
        #print("Similarity Node {}, length of similarity node {}, length of reason {} ".format(similarity_nodes, len(similarity_nodes), len(reason)))

        try:
            score = (len(similarity_nodes) / len(reason)) * 100 

            if(score > 60):
                return [score, 1]
            else:
                return [score, 0]
        except:
            return [0, 0]