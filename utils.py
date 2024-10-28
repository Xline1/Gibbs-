from typing import List, Dict
import numpy as np

def calculate_consensus(sequences: List[str], positions: List[int], motif_length: int) -> str:
    """计算共识序列"""
    count_matrix = {'A': [0] * motif_length,
                   'C': [0] * motif_length,
                   'G': [0] * motif_length,
                   'T': [0] * motif_length}
    
    for seq, pos in zip(sequences, positions):
        motif = seq[pos:pos + motif_length]
        for i, base in enumerate(motif):
            count_matrix[base][i] += 1
    
    consensus = ''
    for i in range(motif_length):
        max_base = max(count_matrix.keys(), 
                      key=lambda x: count_matrix[x][i])
        consensus += max_base
    
    return consensus

def evaluate_motif_predictions(predicted_positions: List[int], true_positions: List[int], 
                             sequences: List[str], motif_length: int) -> Dict:
    """
    评估预测的模体位置
    
    参数:
        predicted_positions: 预测的模体起始位置列表
        true_positions: 真实的模体起始位置列表
        sequences: DNA序列列表
        motif_length: 模体长度
    
    返回:
        包含评估结果的字典
    """
    if len(predicted_positions) != len(true_positions):
        raise ValueError("预测位置和真实位置的数量不匹配")
    
    total_sequences = len(sequences)
    sequence_scores = []  # 存储每个序列的匹配得分
    
    for i in range(total_sequences):
        # 获取预测的模体和真实的模体
        pred_start = predicted_positions[i]
        true_start = true_positions[i]
        
        pred_motif = sequences[i][pred_start:pred_start + motif_length]
        true_motif = sequences[i][true_start:true_start + motif_length]
        
        # 计算碱基匹配数
        matches = sum(1 for p, t in zip(pred_motif, true_motif) if p == t)
        sequence_score = matches / motif_length
        sequence_scores.append(sequence_score)
    
    # 计算总体统计
    average_score = np.mean(sequence_scores)
    perfect_matches = sum(1 for score in sequence_scores if score == 1.0)
    
    return {
        'sequence_scores': sequence_scores,  # 每个序列的匹配得分
        'average_score': average_score,      # 平均匹配得分
        'perfect_matches': perfect_matches,  # 完全匹配的数量
        'perfect_ratio': perfect_matches / total_sequences,  # 完全匹配的比例
        'total_sequences': total_sequences   # 总序列数
    }
