import numpy as np
from typing import List, Tuple
import math

class GibbsSampler:
    def __init__(self, sequences: List[str], motif_length: int, pseudocount: float = 1.0):
        self.sequences = sequences
        self.motif_length = motif_length
        self.pseudocount = pseudocount
        self.n_sequences = len(sequences)
        self.base_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        
        # 验证序列长度和模体长度的合法性
        for seq in sequences:
            if len(seq) < motif_length:
                raise ValueError(f"序列长度({len(seq)})小于模体长度({motif_length})")
        
        # 初始化位置和最佳结果
        self.positions = self._initialize_positions()
        self.best_positions = self.positions.copy()
        self.best_score = float('-inf')
        
        self.prev_pssm = None  # 添加用于存储上一次的PSSM矩阵
    
    def _initialize_positions(self) -> List[int]:
        """随机初始化模体位置"""
        positions = []
        for seq in self.sequences:
            max_pos = len(seq) - self.motif_length
            positions.append(np.random.randint(0, max_pos + 1))
        return positions
    
    def _build_pfm(self, exclude_seq: int = None) -> np.ndarray:
        """构建位置频数矩阵(PFM)"""
        pfm = np.zeros((4, self.motif_length)) + self.pseudocount  # 添加伪计数
        
        for i, (seq, pos) in enumerate(zip(self.sequences, self.positions)):
            if i == exclude_seq:
                continue
            motif = seq[pos:pos + self.motif_length]
            for j, base in enumerate(motif):
                pfm[self.base_to_index[base]][j] += 1
                
        return pfm
    
    def _build_ppm(self, pfm: np.ndarray) -> np.ndarray:
        """构建位置概率矩阵(PPM)"""
        return pfm / pfm.sum(axis=0)
    
    def _build_pssm(self, exclude_seq: int = None) -> np.ndarray:
        """构建位点特异性打分矩阵(PSSM)"""
        pfm = self._build_pfm(exclude_seq)
        ppm = self._build_ppm(pfm)
        background = 0.25  # 背景概率
        pssm = np.log2(ppm / background)  # 使用log2计算得分
        return pssm
    
    def calculate_score(self, sequence: str, position: int, pssm: np.ndarray) -> float:
        """计算给定位置的得分"""
        score = 0
        subsequence = sequence[position:position + self.motif_length]
        if len(subsequence) != self.motif_length:
            return float('-inf')  # 避免边界问题
        
        # 添加位置权重
        position_weight = 1.0
        if position == 0 or position == len(sequence) - self.motif_length:
            position_weight = 0.8  # 降低序列两端的权重
        
        for i, base in enumerate(subsequence):
            score += pssm[self.base_to_index[base]][i]
        
        return score * position_weight
    
    def _calculate_total_score(self) -> float:
        """计算当前模体组合的总得分"""
        pssm = self._build_pssm()
        total_score = 0
        for seq, pos in zip(self.sequences, self.positions):
            total_score += self.calculate_score(seq, pos, pssm)
        return total_score
    
    def _is_converged(self, current_pssm: np.ndarray, tolerance: float = 1e-6) -> bool:
        """检查PSSM矩阵是否收敛"""
        if self.prev_pssm is None:
            return False
        # 计算两个PSSM矩阵之间的差异
        diff = np.abs(current_pssm - self.prev_pssm).max()
        return diff < tolerance
    
    def calculate_likelihood(self) -> float:
        """计算当前模型的对数似然度"""
        # 1. 构建PWM
        pfm = self._build_pfm()
        pwm = pfm / (self.n_sequences + 4 * self.pseudocount)
        
        # 2. 计算总对数似然度
        total_log_likelihood = 0
        background_prob = 0.25  # 背景概率
        
        for seq_idx, (sequence, pos) in enumerate(zip(self.sequences, self.positions)):
            motif = sequence[pos:pos + self.motif_length]
            motif_log_likelihood = 0
            
            # 计算单个模体的对数似然度
            for j, base in enumerate(motif):
                base_idx = self.base_to_index[base]
                likelihood_ratio = pwm[base_idx, j] / background_prob
                motif_log_likelihood += np.log2(likelihood_ratio)
            
            total_log_likelihood += motif_log_likelihood
        
        return total_log_likelihood
    
    def run(self, max_iterations: int = 10000, n_starts: int = 10, tolerance: float = 1e-6):
        """运行Gibbs采样算法"""
        for start in range(n_starts):
            self.positions = self._initialize_positions()
            prev_likelihood = float('-inf')
            
            for iteration in range(max_iterations):
                # 1. 随机选择一个序列
                seq_idx = np.random.randint(0, self.n_sequences)
                sequence = self.sequences[seq_idx]
                
                # 2. 构建PSSM并更新位置
                current_pssm = self._build_pssm(exclude_seq=seq_idx)
                possible_positions = len(sequence) - self.motif_length + 1
                scores = np.zeros(possible_positions)
                
                for pos in range(possible_positions):
                    scores[pos] = self.calculate_score(sequence, pos, current_pssm)
                
                # 3. 计算位置概率并采样
                scores = scores - np.max(scores)
                probabilities = np.exp(scores)
                probabilities = probabilities / probabilities.sum()
                self.positions[seq_idx] = np.random.choice(possible_positions, p=probabilities)
                
                # 4. 计算新的似然度并检查收敛
                current_likelihood = self.calculate_likelihood()
                if abs(current_likelihood - prev_likelihood) < tolerance:
                    print(f"第{start+1}次重启在第{iteration+1}次迭代后收敛")
                    print(f"最终似然度: {current_likelihood:.2f}")
                    break
                
                prev_likelihood = current_likelihood
            
            # 评估当前解
            current_score = self.calculate_likelihood()
            if current_score > self.best_score:
                self.best_score = current_score
                self.best_positions = self.positions.copy()
        
        # 使用最佳解作为最终结果
        self.positions = self.best_positions
        final_pssm = self._build_pssm()
        return self.positions, final_pssm
