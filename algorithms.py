from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple

class MotifFinder(ABC):
    """模体查找算法的基类"""
    def __init__(self, sequences: List[str], motif_length: int):
        self.sequences = sequences
        self.motif_length = motif_length
        self.n_sequences = len(sequences)
        self.base_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        
    @abstractmethod
    def find_motif(self) -> Tuple[List[int], np.ndarray]:
        """查找模体，返回位置列表和PSSM矩阵"""
        pass
    
    def _build_ppm(self, pfm: np.ndarray) -> np.ndarray:
        """构建位置概率矩阵(PPM)"""
        return pfm / pfm.sum(axis=0)
    
    def _build_pssm(self, ppm: np.ndarray) -> np.ndarray:
        """构建位点特异性打分矩阵(PSSM)"""
        background = 0.25  # 背景概率
        return np.log2(ppm / background)

class ExpectationMaximization(MotifFinder):
    """EM算法实现"""
    def __init__(self, sequences: List[str], motif_length: int, pseudocount: float = 1.0):
        super().__init__(sequences, motif_length)
        self.pseudocount = pseudocount
        
    def find_motif(self, max_iterations: int = 100, tolerance: float = 1e-6) -> Tuple[List[int], np.ndarray]:
        # 随机初始化位置
        positions = [np.random.randint(0, len(seq) - self.motif_length + 1) 
                    for seq in self.sequences]
        
        # 初始化概率矩阵
        pfm = np.zeros((4, self.motif_length)) + self.pseudocount
        for seq, pos in zip(self.sequences, positions):
            motif = seq[pos:pos + self.motif_length]
            for i, base in enumerate(motif):
                pfm[self.base_to_index[base]][i] += 1
        
        prev_ppm = None
        
        for _ in range(max_iterations):
            # E步：计算每个位置的概率
            ppm = self._build_ppm(pfm)
            new_positions = []
            new_pfm = np.zeros((4, self.motif_length)) + self.pseudocount
            
            for seq_idx, seq in enumerate(self.sequences):
                # 计算每个位置的概率
                probs = np.zeros(len(seq) - self.motif_length + 1)
                for pos in range(len(probs)):
                    prob = 1.0
                    motif = seq[pos:pos + self.motif_length]
                    for i, base in enumerate(motif):
                        prob *= ppm[self.base_to_index[base]][i]
                    probs[pos] = prob
                
                # 归一化概率
                probs = probs / probs.sum()
                
                # 选择最可能的位置
                best_pos = np.argmax(probs)
                new_positions.append(best_pos)
                
                # 更新PFM
                motif = seq[best_pos:best_pos + self.motif_length]
                for i, base in enumerate(motif):
                    new_pfm[self.base_to_index[base]][i] += 1
            
            # 检查收敛
            if prev_ppm is not None:
                if np.abs(new_pfm - prev_ppm).max() < tolerance:
                    break
            
            prev_ppm = new_pfm.copy()
            pfm = new_pfm
            positions = new_positions
        
        return positions, self._build_pssm(self._build_ppm(pfm))

class MEME(MotifFinder):
    """MEME算法实现"""
    def __init__(self, sequences: List[str], motif_length: int, pseudocount: float = 1.0):
        super().__init__(sequences, motif_length)
        self.pseudocount = pseudocount
        
    def find_motif(self, max_iterations: int = 100, tolerance: float = 1e-6) -> Tuple[List[int], np.ndarray]:
        # 初始化
        best_positions = None
        best_score = float('-inf')
        
        # 尝试每个k-mer作为起始点
        for seq_idx, seq in enumerate(self.sequences):
            for pos in range(len(seq) - self.motif_length + 1):
                # 使用当前k-mer构建初始模型
                pfm = np.zeros((4, self.motif_length)) + self.pseudocount
                motif = seq[pos:pos + self.motif_length]
                for i, base in enumerate(motif):
                    pfm[self.base_to_index[base]][i] += 1
                
                # EM迭代优化
                positions = [0] * self.n_sequences
                positions[seq_idx] = pos
                prev_ppm = None
                
                for _ in range(max_iterations):
                    ppm = self._build_ppm(pfm)
                    new_pfm = np.zeros((4, self.motif_length)) + self.pseudocount
                    
                    # 对每个序列找最佳匹配位置
                    for i, other_seq in enumerate(self.sequences):
                        if i == seq_idx:
                            continue
                            
                        # 计算每个位置的得分
                        scores = np.zeros(len(other_seq) - self.motif_length + 1)
                        for p in range(len(scores)):
                            score = 0
                            subseq = other_seq[p:p + self.motif_length]
                            for j, base in enumerate(subseq):
                                score += np.log2(ppm[self.base_to_index[base]][j] / 0.25)
                            scores[p] = score
                        
                        # 选择最佳位置
                        best_pos = np.argmax(scores)
                        positions[i] = best_pos
                        
                        # 更新PFM
                        motif = other_seq[best_pos:best_pos + self.motif_length]
                        for j, base in enumerate(motif):
                            new_pfm[self.base_to_index[base]][j] += 1
                    
                    # 检查收敛
                    if prev_ppm is not None:
                        if np.abs(new_pfm - prev_ppm).max() < tolerance:
                            break
                    
                    prev_ppm = new_pfm.copy()
                    pfm = new_pfm
                
                # 计算当前模型的得分
                score = sum(scores)
                if score > best_score:
                    best_score = score
                    best_positions = positions.copy()
        
        # 使用最佳位置构建最终的PSSM
        final_pfm = np.zeros((4, self.motif_length)) + self.pseudocount
        for seq, pos in zip(self.sequences, best_positions):
            motif = seq[pos:pos + self.motif_length]
            for i, base in enumerate(motif):
                final_pfm[self.base_to_index[base]][i] += 1
                
        return best_positions, self._build_pssm(self._build_ppm(final_pfm))
