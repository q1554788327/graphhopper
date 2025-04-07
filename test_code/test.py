import heapq
import networkx as nx
import osmnx as ox

class HighwayStorage:
    def __init__(self, place, network_type='drive'):
        """
        初始化并存储指定地点的OSM highway（道路）信息

        参数:
        - place: 地点名称或区域边界
        - network_type: 网络类型，默认使用驾车道路 ('drive')
        """
        self.place = place
        self.network_type = network_type
        # 下载指定地点的OSM道路网络数据
        self.graph = ox.graph_from_place(place, network_type=network_type)
        # 解析图中的道路边，存储在字典中
        self.highways = self._parse_highways()
    
    def _parse_highways(self):
        """
        解析OSM图中的每条道路边，并提取相关属性（如道路类别、速度限制、是否单向、长度等）
        返回:
            highways: 字典，键为 edge_id（组合 u-v-key），值为包含道路属性的字典
        """
        highways = {}
        # 遍历 MultiDiGraph 中所有的边（包含多条平行边）
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            edge_id = f"{u}-{v}-{key}"
            highway_type = data.get('highway', None)
            maxspeed = data.get('maxspeed', None)
            # oneway 属性通常指示该路段是否为单向，默认为 False
            oneway = data.get('oneway', False)
            highways[edge_id] = {
                'u': u,
                'v': v,
                'highway': highway_type,
                'maxspeed': maxspeed,
                'oneway': oneway,
                'length': data.get('length', None)
            }
        return highways

    def get_edge(self, edge_id):
        """
        根据edge_id获取道路边属性
        """
        return self.highways.get(edge_id, None)

    def get_all_edges(self):
        """
        返回所有线路边的信息
        """
        return self.highways

class Weighting:
    def __init__(self, turn_penalty=5):
        self.turn_penalty = turn_penalty

    def calc_edge_weight(self, edge_data, reverse=False):
        """计算边的基本权重，假设边数据中有 'weight' 属性"""
        return edge_data.get('weight', 1)

    def calc_turn_weight(self, prev_mode, current_mode):
        """
        计算转弯（模式切换）权重：
        - 如果前后模式一致，则无转弯成本；
        - 如果模式不同，则增加一个固定的换乘成本。
        """
        if prev_mode is None or prev_mode == current_mode:
            return 0
        return self.turn_penalty

def calc_weight_with_turn_weight(weighting, edge_data, reverse, prev_mode):
    """
    计算实际边权重：基本边权重 + 转弯权重
    当 prev_mode 为 None 时，表示起始状态，不增加转弯成本。
    """
    edge_weight = weighting.calc_edge_weight(edge_data, reverse)
    current_mode = edge_data.get('mode')
    if prev_mode is None:
        return edge_weight
    turn_weight = weighting.calc_turn_weight(prev_mode, current_mode)
    return edge_weight + turn_weight

class MultiModalPathPlanner:
    def __init__(self, graph, weighting):
        # 使用 networkx MultiDiGraph 来支持多条边（不同模式）
        self.graph = graph
        self.weighting = weighting

    def calc_path(self, start, goal):
        """
        采用 Dijkstra 算法计算最优路径。
        状态包含：(累计权重, 当前节点, 前一条使用的模式, 路径列表)
        """
        # 初始化优先队列：起始节点的前一模式设置为 None
        queue = []
        heapq.heappush(queue, (0, start, None, [start]))
        best = {(start, None): 0}
        
        while queue:
            cost, node, prev_mode, path = heapq.heappop(queue)
            if node == goal:
                return path, cost
            for neighbor in self.graph.neighbors(node):
                # MultiDiGraph 可能有多条边连接同一对节点
                for edge_key, edge_data in self.graph.get_edge_data(node, neighbor).items():
                    additional_weight = calc_weight_with_turn_weight(self.weighting, edge_data, False, prev_mode)
                    new_cost = cost + additional_weight
                    new_mode = edge_data.get('mode')
                    state = (neighbor, new_mode)
                    if state not in best or new_cost < best[state]:
                        best[state] = new_cost
                        new_path = path + [neighbor]
                        heapq.heappush(queue, (new_cost, neighbor, new_mode, new_path))
        return None, float('inf')

if __name__ == "__main__":
    # 构造一个简单的多模式图：
    # 节点：1、2、3、4
    # 边：
    #   1->2: 步行 ("walk") 边，权重 2
    #   2->3: 公交 ("transit") 边，权重 3
    #   1->3: 步行 ("walk") 边，权重 5
    #   3->4: 公交 ("transit") 边，权重 2
    #   2->4: 步行 ("walk") 边，权重 10
    G = nx.MultiDiGraph()
    # 添加步行边
    G.add_edge(1, 2, key=0, weight=2, mode='walk')
    G.add_edge(1, 3, key=0, weight=5, mode='walk')
    G.add_edge(2, 4, key=0, weight=10, mode='walk')
    
    # 添加公交边
    G.add_edge(2, 3, key=0, weight=3, mode='transit')
    G.add_edge(3, 4, key=0, weight=2, mode='transit')
    
    # 设置转乘换模式的额外成本
    weighting = Weighting(turn_penalty=5)
    planner = MultiModalPathPlanner(G, weighting)
    
    start_node = 1
    goal_node = 4
    path, total_cost = planner.calc_path(start_node, goal_node)
    print("Path:", path)
    print("Total cost:", total_cost)