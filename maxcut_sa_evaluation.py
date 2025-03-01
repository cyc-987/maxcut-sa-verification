import numpy as np
import math
import random
import torch
import sys
import tqdm

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
 
class SA:
    def __init__(self, 
                 question: Question, # 问题
                 T_initial: int=1000, # 初始温度
                 alpha: float=0.999, # 降温系数
                 iter: int=1000): # 迭代次数
        '''
        初始化模拟退火算法 \n
        question: 问题 \n
        T_initial: 初始温度 \n
        alpha: 降温系数 \n
        iter: 迭代次数
        '''
        # 参数
        self.question = question
        self.temp_initial = T_initial
        self.alpha = alpha
        self.iter = iter
        self.device = self.question.device
        
        # 历史求解结果记录
        self.solve_times = 0
        self.solution_energy = []
        self.solution_cut_value = []
        
        print(f"SA algorithm initialized, initial temperature: {self.temp_initial}, alpha: {self.alpha}, iteration: {self.iter}.") 
    
    def _GenerateNewX(self,
                      X,
                      type: str="flip",
                      flip_num: int=1):
        '''
        生成新解，同时检查旧解和新解的合法性 \n
        X: 当前解
        type: 生成新解的方式，flip表示翻转 \n
        flip_num: 翻转的个数
        '''
        self._CheckX(X)
        
        flip_num = min(flip_num, len(X))
        indices = random.sample(range(len(X)), flip_num)
        X_new = X.clone()
        X_new[indices] *= -1
        
        self._CheckX(X_new)
        
        return X_new, indices
    
    def _CalDeltaE (self, X, X_new, flip_index):
        '''
        计算能量差 \n
        X: 当前解 \n
        X_new: 新解 \n
        flip_index: 翻转的索引
        '''
        X_row = X.clone()
        X_row[flip_index] = 0 
        delta_E = 0 
        
        for i in flip_index:
            # 提取 J 的 i 列
            J_col = self.question.matrix[:, i]
            # 计算 X_new 和 J_col 的乘积，得到新向量
            new_J_col = J_col * X_new[i]      
            delta_E = delta_E + torch.dot(X_row, new_J_col) 
        
        return delta_E
        
    
    def Solve(self, output: bool=True):
        '''
        求解 \n
        output: 是否输出结果（包括tqdm进度条）
        '''
        temp = self.temp_initial
        # 生成历史记录数组
        energy_history = []
        cut_value_history = []
        # 生成初始解（全1）
        X = np.ones(self.question.node_num)
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        self._CheckX(X)
        
        # 计算能量
        energy = torch.dot(X, torch.mv(self.question.matrix, X))
        # 计算cut值
        cut_value = torch.sum(torch.triu(self.question.matrix, diagonal=1) * 0.5 * (1 - torch.ger(X, X)))
        # 降温
        temp *= self.alpha
        
        # 记录历史
        energy_history.append(energy.item())
        cut_value_history.append(cut_value.item())
        if(output):print(f"Initial energy: {energy.item()}, initial cut value: {cut_value.item()}.")
        
        # 迭代
        for iteration in tqdm.tqdm(range(self.iter), disable=not output):
            # 生成新解
            X_new, flip_index = self._GenerateNewX(X, flip_num=2)
            delta_E = self._CalDeltaE(X, X_new, flip_index)
            
            # 接受解
            if delta_E <= 0:
                X = X_new
            else:
                random_number = random.uniform(0, 1)
                if random_number < math.exp(-delta_E.item() / temp):
                    X = X_new
                else:
                    pass
            
            # 后计算
            energy = torch.dot(X, torch.mv(self.question.matrix, X))
            cut_value = torch.sum(torch.triu(self.question.matrix, diagonal=1) * 0.5 * (1 - torch.ger(X, X)))
            temp *= self.alpha
            
            # 记录历史
            energy_history.append(energy.item())
            cut_value_history.append(cut_value.item())
        
        if(output):print(f"Iteration finished, total iteration: {self.iter}.")
        if(output):print(f"Final energy: {energy.item()}, final cut value: {cut_value.item()}.")
        self.solve_times += 1
        self.solution_energy.append(energy.item())
        self.solution_cut_value.append(cut_value.item())
        
    def MultiSolver(self, 
                    times: int=2):
        '''
        多次求解 \n
        times: 求解次数
        '''
        print(f"Start multi-solving, total times: {times}.")
        for iteration in tqdm.tqdm(range(times)):
            self.Solve(output=False)
        
        # 找到最优解
        best_energy = min(self.solution_energy)
        best_energy_index = self.solution_energy.index(best_energy)
        best_cut_value = self.solution_cut_value[best_energy_index]
        print(f"Multi-solving finished, best energy: {best_energy}, best cut value: {best_cut_value}, index: {best_energy_index}.")
        
        
        
    def _CheckX(self, X):
        '''
        检查X中元素是否只包含1或-1
        '''
        if not torch.all((X == 1) | (X == -1)):
            print(f"Before CiM: X wrong!")
            invalid_mask = ~((X == 1) | (X == -1))
            # 提取这些元素的位置和值
            invalid_indices = torch.where(invalid_mask)[0]
            invalid_values = X[invalid_mask]
            # 打印结果
            print("Invalid indices:", invalid_indices)
            print("Invalid values:", invalid_values)        
            print(f"\n")
            sys.exit()


if __name__ == '__main__':
    index = 1
    problem = Question(index)
    
    sa = SA(problem, iter=1000)
    sa.Solve()
    sa.MultiSolver(10)