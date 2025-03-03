import numpy as np
import torch
import sys

class Question:
    def __init__(self, 
                 problemset_index: int):
        '''
        初始化问题 \n
        problemset_index: 问题集编号
        '''
        # 环境变量
        self.filepath_head = "./gset/yyye/Gset/G"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device type: {self.device}")
        
        # 参数
        self.problemset_index = problemset_index # 问题集编号
        print(f"Index selected: {self.problemset_index}")
        self.path = self.filepath_head + str(self.problemset_index) # 文件路径
        self.node_num = 0 # 节点数
        self.edge_num = 0 # 边数
        self.matrix = None # 邻接矩阵
        self.matrix_active_num_check = 0 # 邻接矩阵节点数检查
        
        # 读文件
        self._ReadFileContent()
        # 检查邻接矩阵
        self._CheckMatrix()
        
    def _ReadFileContent(self):
        '''
        读取文件内容
        '''
        with open(self.path, "r") as f:
            # 读取第一行，获取节点数和边数
            firstline = f.readline()
            self.node_num, self.edge_num = firstline.split()
            self.node_num = int(self.node_num)
            self.edge_num = int(self.edge_num)
            
            # 初始化邻接矩阵
            self.matrix = np.zeros((int(self.node_num), int(self.node_num)))
            lines = f.read().splitlines()
            lines_num = len(lines)
            for i in range(int(lines_num)):
                aline = lines[i]
                node1, node2, weight = aline.split()
                self.matrix[int(node1)-1][int(node2)-1] = int(weight)
                self.matrix[int(node2)-1][int(node1)-1] = int(weight)
            
            self.matrix = torch.tensor(self.matrix, dtype=torch.float32).to(self.device)
            print(f"Read file \"{self.path}\" successfully, read {self.node_num} nodes and {self.edge_num} edges.")
 
    def _CheckMatrix(self):
        '''
        检查邻接矩阵是否正确
        '''
        active_num = torch.sum(self.matrix != 0).item()
        self.matrix_active_num_check = active_num
        print(f"Check matrix: active number: {active_num}.")
        
        # 检查对称性
        if not torch.all(self.matrix == self.matrix.t()):
            print(f"Check matrix: not symmetric!")
            sys.exit()
        
        # 检查对角线
        if torch.any(torch.diagonal(self.matrix) != 0):
            print(f"Check matrix: diagonal wrong!")
            sys.exit()
        
        print("Check matrix: correct.")
 