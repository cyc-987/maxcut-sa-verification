import numpy as np
import math
import random
import torch
import sys
import tqdm
from Question import Question
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
        count = 0
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
        count += 1
        
        # 记录历史
        energy_history.append(energy.item())
        cut_value_history.append(cut_value.item())
        if(output):print("----------Solve Start----------")
        if(output):print(f"Initial energy: {energy.item()}, initial cut value: {cut_value.item()}.")
        
        # 迭代
        for iteration in tqdm.tqdm(range(self.iter), disable=not output):
            # 生成新解
            accept = True
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
                    accept = False
            
            count += 1
            temp = self._RecordHistory(X, energy_history, cut_value_history, temp, accept)
        
        if(output):print(f"Iteration finished, total number of generated solutions: {count}.")
        if(output):print(f"Final energy: {energy_history[-1]}, final cut value: {cut_value_history[-1]}.")
        if(output):print("----------Solve Finished----------")
        self.solve_times += 1
        self.solution_energy.append(energy_history[-1])
        self.solution_cut_value.append(cut_value_history[-1])
        
        return energy_history, cut_value_history
    
    def _RecordHistory(self, X, 
                       energy_history: list, cut_value_history: list, 
                       temp: float, update: bool):
        '''
        记录历史 \n
        返回值：新温度（降温后）
        '''
        if update:
        # 后计算
            energy = torch.dot(X, torch.mv(self.question.matrix, X))
            cut_value = torch.sum(torch.triu(self.question.matrix, diagonal=1) * 0.5 * (1 - torch.ger(X, X)))
            
            # 记录历史
            energy_history.append(energy.item())
            cut_value_history.append(cut_value.item())
            
        temp *= self.alpha
        return temp
    
    def ParallelSolve(self, 
                      t1: float=0.8, 
                      t2: float=0.5, 
                      output: bool=True):
        '''
        并行求解 \n
        策略：温度大于t1，四路并行赌博求解；温度在t2和t1之间，两路并行赌博求解；温度小于t2，串行赌博求解 \n
        t1: 温度阈值1 \n
        t2: 温度阈值2 \n
        t1和t2均为0-1之间的小数，代表相对于初始温度的比例
        '''
        temp1 = self.temp_initial * t1
        temp2 = self.temp_initial * t2

        count = 0
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
        count += 1
        
        # 记录历史
        energy_history.append(energy.item())
        cut_value_history.append(cut_value.item())
        if(output):print("----------Parallel Solve Start----------")
        if(output):print(f"Initial energy: {energy.item()}, initial cut value: {cut_value.item()}.")

        # 迭代
        for iteration in tqdm.tqdm(range(self.iter), disable=not output):
            # 生成新解
            if temp > temp1:
                X_new_1, flip_index_1 = self._GenerateNewX(X, flip_num=2)
                X_new_2, flip_index_2 = self._GenerateNewX(X_new_1, flip_num=2)
                X_new_3, flip_index_3 = self._GenerateNewX(X_new_2, flip_num=2)
                X_new_4, flip_index_4 = self._GenerateNewX(X_new_3, flip_num=2)
                
                end = False # 结束标志
                # 无条件检测第一个解
                delta_E = self._CalDeltaE(X, X_new_1, flip_index_1)
                if delta_E <= 0:
                    X = X_new_1
                else:
                    random_number = random.uniform(0, 1)
                    if random_number < math.exp(-delta_E.item() / temp):
                        X = X_new_1
                    else: end = True # 如果不接受，丢弃后面全部解
                temp = self._RecordHistory(X, energy_history, cut_value_history, temp, not end)
                count += 1
                
                # 检测第二个解
                if not end:
                    delta_E = self._CalDeltaE(X, X_new_2, flip_index_2)
                    if delta_E <= 0:
                        X = X_new_2
                    else:
                        random_number = random.uniform(0, 1)
                        if random_number < math.exp(-delta_E.item() / temp):
                            X = X_new_2
                        else: end = True
                    temp = self._RecordHistory(X, energy_history, cut_value_history, temp, not end)
                    count += 1
                
                # 检测第三个解
                if not end:
                    delta_E = self._CalDeltaE(X, X_new_3, flip_index_3)
                    if delta_E <= 0:
                        X = X_new_3
                    else:
                        random_number = random.uniform(0, 1)
                        if random_number < math.exp(-delta_E.item() / temp):
                            X = X_new_3
                        else: end = True
                    temp = self._RecordHistory(X, energy_history, cut_value_history, temp, not end)
                    count += 1
                
                # 检测第四个解
                if not end:
                    delta_E = self._CalDeltaE(X, X_new_4, flip_index_4)
                    if delta_E <= 0:
                        X = X_new_4
                    else:
                        random_number = random.uniform(0, 1)
                        if random_number < math.exp(-delta_E.item() / temp):
                            X = X_new_4
                        else: end = True
                    temp = self._RecordHistory(X, energy_history, cut_value_history, temp, not end)
                    count += 1
                
            
            elif temp <= temp1 and temp > temp2:
                pass
                # TBD
        
        
        
        if(output):print(f"Iteration finished, total number of generated solutions: {count}.")
        if(output):print(f"Final energy: {energy_history[-1]}, final cut value: {cut_value_history[-1]}.")
        if(output):print("----------Parallel Solve Finished----------")
        self.solve_times += 1
        self.solution_energy.append(energy_history[-1])
        self.solution_cut_value.append(cut_value_history[-1])
        
        return energy_history, cut_value_history

        
        
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
    # sa.MultiSolver(10)
    sa.ParallelSolve(t1=0)