import numpy as np
from typing import List, Tuple

def generate_test_sequences(count: int, length: int, motif_length: int, mutation_prob: float = 0.1):
    """生成测试序列"""
    sequences = []
    true_positions = []
    bases = ['A', 'C', 'G', 'T']
    
    # 生成随机模体
    motif = ''.join(np.random.choice(bases) for _ in range(motif_length))
    
    # 确保序列长度合适
    if length < motif_length * 2:
        length = motif_length * 2
    
    for _ in range(count):
        # 生成随机背景序列
        sequence = ''.join(np.random.choice(bases) for _ in range(length))
        sequence = list(sequence)
        
        # 随机选择插入位置
        max_position = length - motif_length
        insert_position = np.random.randint(0, max_position + 1)
        true_positions.append(insert_position)
        
        # 插入可能发生突变的模体
        mutated_motif = list(motif)
        for i in range(len(motif)):
            if np.random.random() < mutation_prob:
                # 随机选择一个不同的碱基进行突变
                available_bases = [b for b in bases if b != motif[i]]
                mutated_motif[i] = np.random.choice(available_bases)
        
        # 将模体插入序列
        sequence[insert_position:insert_position + motif_length] = mutated_motif
        sequences.append(''.join(sequence))
    
    return sequences, true_positions, motif  # 返回原始模体
