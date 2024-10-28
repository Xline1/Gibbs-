import tkinter as tk
from tkinter import ttk, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import logomaker
import numpy as np
import time
from typing import List

from gibbs_sampler import GibbsSampler
from sequence_generator import generate_test_sequences
from utils import calculate_consensus, evaluate_motif_predictions
# 添加新的导入
from algorithms import ExpectationMaximization, MEME

class GibbsSamplerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Gibbs采样识别模体")  # 更新标题
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建主框架
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 初始化界面组件
        self.create_sequence_generator()
        self.create_input_area()
        self.create_control_buttons()
        self.create_result_area()
        self.create_visualization_area()
    
    # ... [其余GUI法保持不变] ...

    def create_sequence_generator(self):
        """创建序列生成器区域"""
        gen_frame = ttk.LabelFrame(self.main_frame, text="序列生成器", padding="5")
        gen_frame.grid(row=0, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        # 序列数量
        ttk.Label(gen_frame, text="序列数量:").grid(row=0, column=0, sticky=tk.W)
        self.seq_count = ttk.Entry(gen_frame, width=10)
        self.seq_count.insert(0, "5")
        self.seq_count.grid(row=0, column=1, padx=5)
        
        # 序列长度
        ttk.Label(gen_frame, text="序列长度:").grid(row=0, column=2, sticky=tk.W)
        self.seq_length = ttk.Entry(gen_frame, width=10)
        self.seq_length.insert(0, "50")
        self.seq_length.grid(row=0, column=3, padx=5)
        
        # 植入模体
        ttk.Label(gen_frame, text="植入模体:").grid(row=1, column=0, sticky=tk.W)
        self.motif_pattern = ttk.Entry(gen_frame, width=20)
        self.motif_pattern.insert(0, "ACGTACGTAC")
        self.motif_pattern.grid(row=1, column=1, columnspan=2, padx=5)
        
        # 生成按钮
        ttk.Button(gen_frame, text="生成序列", command=self.generate_sequences).grid(row=1, column=3, padx=5)

    def create_input_area(self):
        """创建输入区域"""
        # 创建参数框架
        param_frame = ttk.LabelFrame(self.main_frame, text="参数设置", padding="5")
        param_frame.grid(row=0, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        # 序列数量
        ttk.Label(param_frame, text="序列数量:").grid(row=0, column=0, sticky=tk.W)
        self.seq_count = ttk.Entry(param_frame, width=10)
        self.seq_count.insert(0, "5")
        self.seq_count.grid(row=0, column=1, padx=5)
        
        # 序列长度
        ttk.Label(param_frame, text="序列长度:").grid(row=0, column=2, sticky=tk.W)
        self.seq_length = ttk.Entry(param_frame, width=10)
        self.seq_length.insert(0, "50")
        self.seq_length.grid(row=0, column=3, padx=5)
        
        # 模体长度
        ttk.Label(param_frame, text="模体长度:").grid(row=1, column=0, sticky=tk.W)
        self.motif_length = ttk.Entry(param_frame, width=10)
        self.motif_length.insert(0, "8")
        self.motif_length.grid(row=1, column=1, padx=5)
        
        # 伪计数
        ttk.Label(param_frame, text="伪计数:").grid(row=1, column=2, sticky=tk.W)
        self.pseudocount = ttk.Entry(param_frame, width=10)
        self.pseudocount.insert(0, "1.0")
        self.pseudocount.grid(row=1, column=3, padx=5)
        
        # 生成序列按钮
        ttk.Button(param_frame, text="生成序列", command=self.generate_sequences).grid(
            row=2, column=0, columnspan=4, pady=5
        )
        
        # 序列显示区域
        self.sequence_text = scrolledtext.ScrolledText(self.main_frame, width=50, height=10)
        self.sequence_text.grid(row=1, column=0, columnspan=2, pady=5)

    def create_control_buttons(self):
        """创建控制按钮"""
        button_frame = ttk.Frame(self.main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=5)
        
        # 运行算法按钮
        ttk.Button(button_frame, text="运行算法", command=self.run_algorithm).pack(side=tk.LEFT, padx=5)
        
        # 参数分析按钮
        ttk.Button(button_frame, text="参数分析", command=self.run_gibbs_analysis).pack(side=tk.LEFT, padx=5)
        
        # 算法对比按钮
        ttk.Button(button_frame, text="算法对比", command=self.run_algorithm_comparison).pack(side=tk.LEFT, padx=5)
        
        # 清除结果按钮
        ttk.Button(button_frame, text="清除结果", command=self.clear_results).pack(side=tk.LEFT, padx=5)

    def create_result_area(self):
        """创建果显示区域"""
        result_frame = ttk.LabelFrame(self.main_frame, text="结果", padding="5")
        result_frame.grid(row=7, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        self.result_text = scrolledtext.ScrolledText(result_frame, width=50, height=10)
        self.result_text.grid(row=0, column=0, pady=5)

    def create_visualization_area(self):
        """创建可视化区域"""
        # 创建更小的图形
        self.fig = Figure(figsize=(6, 1.2))  # 进一步减小高度
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.get_tk_widget().grid(row=8, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))

    def generate_sequences(self):
        """生成测试序列"""
        try:
            count = int(self.seq_count.get())
            length = int(self.seq_length.get())
            motif_length = int(self.motif_length.get())
            
            # 生成序列
            sequences, true_positions, true_motif = generate_test_sequences(count, length, motif_length)
            
            # 显示序列和模体信息
            self.sequence_text.delete("1.0", tk.END)
            self.sequence_text.insert(tk.END, f"插入的模体: {true_motif}\n")
            self.sequence_text.insert(tk.END, "生成的序列:\n")
            for i, seq in enumerate(sequences):
                self.sequence_text.insert(tk.END, f"序列 {i+1} (位置 {true_positions[i]}): {seq}\n")
            
            # 保存信息用于评估
            self.sequences = sequences
            self.true_motif_positions = true_positions
            self.true_motif = true_motif
            
        except ValueError as e:
            self.result_text.delete("1.0", tk.END)
            self.result_text.insert(tk.END, f"错误: {str(e)}")

    def run_algorithm(self):
        """运行算法"""
        try:
            if not hasattr(self, 'sequences') or not self.sequences:
                raise ValueError("请先生成序列")
            
            motif_length = int(self.motif_length.get())
            pseudocount = float(self.pseudocount.get())
            
            # 运行算法
            gibbs = GibbsSampler(self.sequences, motif_length, pseudocount)
            positions, pssm = gibbs.run()
            
            # 评估结果
            eval_results = evaluate_motif_predictions(
                positions, 
                self.true_motif_positions,
                self.sequences,
                motif_length
            )
            
            # 显示结果
            self.result_text.delete("1.0", tk.END)
            self.result_text.insert(tk.END, f"原始模体: {self.true_motif}\n\n")
            self.result_text.insert(tk.END, "预测结果:\n")
            for i, pos in enumerate(positions):
                pred_motif = self.sequences[i][pos:pos + motif_length]
                true_motif = self.sequences[i][self.true_motif_positions[i]:self.true_motif_positions[i] + motif_length]
                self.result_text.insert(tk.END, f"序列 {i+1}:\n")
                self.result_text.insert(tk.END, f"  预测位置: {pos} (模体: {pred_motif})\n")
                self.result_text.insert(tk.END, f"  真实位置: {self.true_motif_positions[i]} (模体: {true_motif})\n")
                self.result_text.insert(tk.END, f"  匹配率: {eval_results['sequence_scores'][i]:.2%}\n")
            
            self.result_text.insert(tk.END, f"\n总体评估:\n")
            self.result_text.insert(tk.END, f"平均匹配率: {eval_results['average_score']:.2%}\n")
            self.result_text.insert(tk.END, f"完全匹配数: {eval_results['perfect_matches']}/{eval_results['total_sequences']}\n")
            
            # 创建并显示序列Logo
            self.create_sequence_logo(self.sequences, positions, motif_length)
            
        except Exception as e:
            self.result_text.delete("1.0", tk.END)
            self.result_text.insert(tk.END, f"运行错误: {str(e)}")

    def create_sequence_logo(self, sequences: List[str], positions: List[int], motif_length: int):
        """创建并显示序列Logo"""
        # 提取模体序列
        motifs = [seq[pos:pos + motif_length] for seq, pos in zip(sequences, positions)]
        
        # 创建计数矩阵
        counts_matrix = np.zeros((4, motif_length))
        for motif in motifs:
            for i, base in enumerate(motif):
                if base in 'ACGT':
                    base_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}[base]
                    counts_matrix[base_idx, i] += 1
        
        # 转换为概率矩阵
        prob_matrix = counts_matrix / len(motifs)
        
        # 创建Logo图
        fig = Figure(figsize=(8, 3))
        ax = fig.add_subplot(111)
        
        # 使用logomaker创建Logo
        df = pd.DataFrame(prob_matrix.T, columns=['A', 'C', 'G', 'T'])
        logo = logomaker.Logo(df, ax=ax)
        
        # 设置图表属性
        ax.set_title('序列Logo图')
        
        # 如果已有canvas，先移除
        if hasattr(self, 'canvas'):
            self.canvas.get_tk_widget().grid_forget()
        
        # 创建新的canvas并显示
        self.canvas = FigureCanvasTkAgg(fig, master=self.main_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=8, column=0, columnspan=2, pady=10)

    def display_results(self, sampler, sequences):
        """显示算法结果"""
        self.result_text.delete("1.0", tk.END)
        
        # 显示算法运行信息
        self.result_text.insert(tk.END, "算运行完成\n\n")
        
        # 显示找到的模体
        self.result_text.insert(tk.END, "找到的模体序列:\n")
        for i, (seq, pos) in enumerate(zip(sequences, sampler.positions)):
            motif = seq[pos:pos + sampler.motif_length]
            self.result_text.insert(tk.END, f"序列 {i+1}: {motif} (位置: {pos})\n")
        
        # 显示共识序列
        consensus = calculate_consensus(sequences, sampler.positions, sampler.motif_length)
        self.result_text.insert(tk.END, f"\n共识序列: {consensus}\n")
        
        # 显示最终得分
        self.result_text.insert(tk.END, f"\n最终得分: {sampler.best_score:.2f}\n")
        
        # 显示性能评估
        if hasattr(self, 'true_motif_positions'):
            results = evaluate_motif_predictions(sampler.positions, self.true_motif_positions)
            accuracy = (results['exact_matches'] + results['close_matches']) / results['total']
            self.result_text.insert(tk.END, f"\n算法性能评估:\n")
            self.result_text.insert(tk.END, f"精确匹配: {results['exact_matches']}/{results['total']}\n")
            self.result_text.insert(tk.END, f"近似匹配: {results['close_matches']}/{results['total']}\n")
            self.result_text.insert(tk.END, f"总准确率: {accuracy:.2%}\n")
        
        # 更新可视化
        self.update_visualization(sampler._build_pssm())

    def update_visualization(self, pssm):
        """更新序列logo可视化"""
        try:
            self.fig.clear()
            # 调整子图的边距和高度比例
            ax = self.fig.add_subplot(111)
            
            # 转换为概率
            probabilities = np.exp2(pssm)
            probabilities = probabilities.T
            
            # 缩小概率值以减小字母高度
            probabilities = probabilities * 0.2  # 将高度缩小为原来的一半
            
            # 创建DataFrame
            df = pd.DataFrame(probabilities, 
                             columns=['A', 'C', 'G', 'T'],
                             index=range(pssm.shape[1]))
            
            # 创建序列logo，调整字体大小和高度
            logo = logomaker.Logo(df,
                                ax=ax,
                                color_scheme='classic',
                                baseline_width=0,
                                show_spines=False,
                                vsep=0.01,
                                width=0.8,        # 减小字母宽度
                                font_name='SimHei',  # 使用中字体
                                stack_order='small_on_top')
            
            # 设置图形属性，调整字体大小
            ax.set_title('模体序列Logo', fontsize=10, fontproperties='SimHei')
            ax.set_xlabel('位置', fontsize=8, fontproperties='SimHei')
            ax.set_ylabel('相对概率', fontsize=8, fontproperties='SimHei')
            ax.set_ylim([0, 1])  # 将y轴范围设置为0-1，而不是0-2
            
            # 设置刻度标签，调整字体大小
            positions = range(pssm.shape[1])
            ax.set_xticks(positions)
            ax.set_xticklabels([str(i+1) for i in positions], fontsize=8)
            ax.tick_params(axis='y', labelsize=8)
            
            # 添加网格线
            ax.grid(True, axis='y', linestyle='--', alpha=0.3)
            
            # 调整布局，减少边距
            self.fig.tight_layout(pad=0.1)  # 进一步减小边距
            self.canvas.draw()
            
        except Exception as e:
            print(f"可视化错误: {str(e)}")
            import traceback
            traceback.print_exc()

    def clear_results(self):
        """清除结果"""
        self.result_text.delete("1.0", tk.END)
        self.fig.clear()
        self.canvas.draw()

    def run_batch_test(self):
        """运行批量测试"""
        try:
            # 获取参数
            test_count = 50  # 测试次数
            count = int(self.seq_count.get())
            length = int(self.seq_length.get())
            motif = self.motif_pattern.get().upper().strip()
            motif_length = int(self.motif_length.get())
            pseudocount = float(self.pseudocount.get())
            
            # 存储结果
            exact_matches = []
            close_matches = []
            scores = []
            
            # 清空结果区域
            self.result_text.delete("1.0", tk.END)
            self.result_text.insert(tk.END, f"正在进行{test_count}次测试...\n\n")
            self.root.update()
            
            # 进行多次测试
            for i in range(test_count):
                # 生成测试序列
                sequences, true_positions = generate_test_sequences(count, length, motif)
                
                # 运行算
                sampler = GibbsSampler(sequences, motif_length, pseudocount)
                positions, _ = sampler.run(max_iterations=10000, n_starts=10)  # 增加迭代数
                
                # 评估结果
                results = evaluate_motif_predictions(positions, true_positions)
                exact_matches.append(results['exact_matches'] / results['total'])
                close_matches.append((results['exact_matches'] + results['close_matches']) / results['total'])
                scores.append(sampler.best_score)
                
                # 更新进度
                self.result_text.delete("1.0", tk.END)
                self.result_text.insert(tk.END, f"正在进行{test_count}次测试... ({i+1}/{test_count})\n\n")
                self.root.update()
            
            # 显示统计结果
            self.display_test_statistics(exact_matches, close_matches, scores)
            
        except Exception as e:
            self.result_text.delete("1.0", tk.END)
            self.result_text.insert(tk.END, f"测试错误: {str(e)}")

    def display_test_statistics(self, exact_matches, close_matches, scores):
        """显示测试统计结果"""
        # 清空图形
        self.fig.clear()
        
        # 创建子图
        gs = self.fig.add_gridspec(2, 2)
        ax1 = self.fig.add_subplot(gs[0, 0])
        ax2 = self.fig.add_subplot(gs[0, 1])
        ax3 = self.fig.add_subplot(gs[1, :])
        
        # 1. 准确率箱线图
        data = [exact_matches, close_matches]
        ax1.boxplot(data, labels=['精确匹配', '近似匹配'])
        ax1.set_title('准确率分布', fontproperties='SimHei', fontsize=10)
        ax1.set_ylabel('准确率', fontproperties='SimHei', fontsize=8)
        ax1.tick_params(axis='both', labelsize=8)
        
        # 2. 得分分布直方图
        ax2.hist(scores, bins=20, edgecolor='black')
        ax2.set_title('得分分布', fontproperties='SimHei', fontsize=10)
        ax2.set_xlabel('得分', fontproperties='SimHei', fontsize=8)
        ax2.set_ylabel('频次', fontproperties='SimHei', fontsize=8)
        ax2.tick_params(axis='both', labelsize=8)
        
        # 3. 准确率随测试次数的变化
        x = range(1, len(exact_matches) + 1)
        ax3.plot(x, exact_matches, label='精确匹配', alpha=0.7)
        ax3.plot(x, close_matches, label='近似匹配', alpha=0.7)
        ax3.set_title('准确率变化趋势', fontproperties='SimHei', fontsize=10)
        ax3.set_xlabel('测试次数', fontproperties='SimHei', fontsize=8)
        ax3.set_ylabel('准确率', fontproperties='SimHei', fontsize=8)
        ax3.legend(prop={'family': 'SimHei', 'size': 8})
        ax3.tick_params(axis='both', labelsize=8)
        ax3.grid(True, linestyle='--', alpha=0.3)
        
        # 调整布局
        self.fig.tight_layout(pad=0.3)
        self.canvas.draw()
        
        # 显示统计数据
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, "测试统计结果:\n\n")
        self.result_text.insert(tk.END, f"精确匹配率:\n")
        self.result_text.insert(tk.END, f"  平均值: {np.mean(exact_matches):.2%}\n")
        self.result_text.insert(tk.END, f"  标准差: {np.std(exact_matches):.2%}\n\n")
        self.result_text.insert(tk.END, f"近似匹配率:\n")
        self.result_text.insert(tk.END, f"  平均值: {np.mean(close_matches):.2%}\n")
        self.result_text.insert(tk.END, f"  标准差: {np.std(close_matches):.2%}\n\n")
        self.result_text.insert(tk.END, f"得分统计:\n")
        self.result_text.insert(tk.END, f"  平均值: {np.mean(scores):.2f}\n")
        self.result_text.insert(tk.END, f"  标准差: {np.std(scores):.2f}\n")

    def run_gibbs_analysis(self):
        """分析Gibbs采样算法的参数影响"""
        try:
            # 基础参数
            length = int(self.seq_length.get())
            motif_length = int(self.motif_length.get())
            pseudocount = float(self.pseudocount.get())
            
            # 参数范围
            sequence_counts = list(range(5, 31, 5))
            iteration_counts = list(range(1000, 20001, 3000))
            restart_counts = list(range(5, 31, 5))
            
            # 创建进度条
            progress_window = tk.Toplevel(self.root)
            progress_window.title("分析进度")
            progress_window.geometry("300x150")
            
            progress_label = ttk.Label(progress_window, text="正在分析...")
            progress_label.pack(pady=10)
            
            progress_bar = ttk.Progressbar(progress_window, length=200, mode='determinate')
            progress_bar.pack(pady=10)
            
            # 计算总步骤数
            total_steps = (len(sequence_counts) + len(iteration_counts) + len(restart_counts)) * 20
            current_step = 0
            
            # 存储结果
            results = {
                'seq_accuracy': [],
                'iter_accuracy': [],
                'restart_accuracy': []
            }
            
            # 1. 分析序列数量的影响
            for count in sequence_counts:
                accuracies = []
                for _ in range(20):
                    sequences, true_positions, _ = generate_test_sequences(count, length, motif_length)
                    gibbs = GibbsSampler(sequences, motif_length, pseudocount)
                    positions, _ = gibbs.run(max_iterations=10000, n_starts=10)
                    results_eval = evaluate_motif_predictions(
                        positions, true_positions, sequences, motif_length
                    )
                    accuracies.append(results_eval['average_score'])
                    
                    # 更新进度
                    current_step += 1
                    progress = (current_step / total_steps) * 100
                    progress_bar['value'] = progress
                    progress_label['text'] = f"分析进度: {progress:.1f}%"
                    progress_window.update()
                    
                results['seq_accuracy'].append(np.mean(accuracies))
            
            # 2. 分析迭代次数的影响
            sequences, true_positions, _ = generate_test_sequences(10, length, motif_length)
            for iterations in iteration_counts:
                accuracies = []
                for _ in range(20):
                    gibbs = GibbsSampler(sequences, motif_length, pseudocount)
                    positions, _ = gibbs.run(max_iterations=iterations, n_starts=10)
                    results_eval = evaluate_motif_predictions(
                        positions, true_positions, sequences, motif_length
                    )
                    accuracies.append(results_eval['average_score'])
                    
                    # 更新进度
                    current_step += 1
                    progress = (current_step / total_steps) * 100
                    progress_bar['value'] = progress
                    progress_label['text'] = f"分析进度: {progress:.1f}%"
                    progress_window.update()
                    
                results['iter_accuracy'].append(np.mean(accuracies))
            
            # 3. 分析重启次数的影响
            for restarts in restart_counts:
                accuracies = []
                for _ in range(20):
                    gibbs = GibbsSampler(sequences, motif_length, pseudocount)
                    positions, _ = gibbs.run(max_iterations=10000, n_starts=restarts)
                    results_eval = evaluate_motif_predictions(
                        positions, true_positions, sequences, motif_length
                    )
                    accuracies.append(results_eval['average_score'])
                    
                    # 更新进度
                    current_step += 1
                    progress = (current_step / total_steps) * 100
                    progress_bar['value'] = progress
                    progress_label['text'] = f"分析进度: {progress:.1f}%"
                    progress_window.update()
                    
                results['restart_accuracy'].append(np.mean(accuracies))
            
            # 关闭进度窗口
            progress_window.destroy()
            
            # 显示分析结果
            self.display_gibbs_analysis(sequence_counts, iteration_counts, restart_counts, results)
            
        except Exception as e:
            self.result_text.delete("1.0", tk.END)
            self.result_text.insert(tk.END, f"分析错误: {str(e)}")

    def display_gibbs_analysis(self, sequence_counts, iteration_counts, restart_counts, results):
        """显示Gibbs参数分析结果"""
        analysis_window = tk.Toplevel(self.root)
        analysis_window.title("Gibbs采样参数分析")
        
        fig = Figure(figsize=(12, 8))
        canvas = FigureCanvasTkAgg(fig, master=analysis_window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])
        
        # 1. 序列数量的影响
        x = np.array(sequence_counts)
        y = np.array(results['seq_accuracy'])
        z = np.polyfit(x, y, 3)
        p = np.poly1d(z)
        x_smooth = np.linspace(x.min(), x.max(), 100)
        ax1.plot(x, y, 'o', label='实际数据')
        ax1.plot(x_smooth, p(x_smooth), '-', label='拟合曲线')
        ax1.set_title('序列数量的影响', fontproperties='SimHei', fontsize=12)
        ax1.set_xlabel('序列数量', fontproperties='SimHei', fontsize=10)
        ax1.set_ylabel('准确率', fontproperties='SimHei', fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax1.legend(prop={'family': 'SimHei', 'size': 9})
        
        # 2. 迭代次数的影响
        x = np.array(iteration_counts)
        y = np.array(results['iter_accuracy'])
        z = np.polyfit(x, y, 3)
        p = np.poly1d(z)
        x_smooth = np.linspace(x.min(), x.max(), 100)
        ax2.plot(x, y, 'o', label='实际数据')
        ax2.plot(x_smooth, p(x_smooth), '-', label='拟合曲线')
        ax2.set_title('迭代次数的影响', fontproperties='SimHei', fontsize=12)
        ax2.set_xlabel('迭代次数', fontproperties='SimHei', fontsize=10)
        ax2.set_ylabel('准确率', fontproperties='SimHei', fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.3)
        ax2.legend(prop={'family': 'SimHei', 'size': 9})
        
        # 3. 重启次数的影响
        x = np.array(restart_counts)
        y = np.array(results['restart_accuracy'])
        z = np.polyfit(x, y, 3)
        p = np.poly1d(z)
        x_smooth = np.linspace(x.min(), x.max(), 100)
        ax3.plot(x, y, 'o', label='实际数据')
        ax3.plot(x_smooth, p(x_smooth), '-', label='拟合曲线')
        ax3.set_title('重启次数的影响', fontproperties='SimHei', fontsize=12)
        ax3.set_xlabel('重启次数', fontproperties='SimHei', fontsize=10)
        ax3.set_ylabel('准确率', fontproperties='SimHei', fontsize=10)
        ax3.grid(True, linestyle='--', alpha=0.3)
        ax3.legend(prop={'family': 'SimHei', 'size': 9})
        
        fig.tight_layout(pad=1.0)

    def run_algorithm_comparison(self):
        """比较Gibbs采样和EM算法的性能"""
        try:
            # 基础参数
            length = int(self.seq_length.get())
            motif_length = int(self.motif_length.get())
            pseudocount = float(self.pseudocount.get())
            
            # 测试参数
            sequence_counts = [5, 10, 15, 20, 25, 30]
            max_iterations = 10000
            n_starts = 10
            test_repeats = 20
            
            # 创建进度条
            progress_window = tk.Toplevel(self.root)
            progress_window.title("对比进度")
            progress_window.geometry("300x150")
            
            progress_label = ttk.Label(progress_window, text="正在比较算法...")
            progress_label.pack(pady=10)
            
            progress_bar = ttk.Progressbar(progress_window, length=200, mode='determinate')
            progress_bar.pack(pady=10)
            
            # 计算总步骤数
            total_steps = len(sequence_counts) * test_repeats
            current_step = 0
            
            # 存储结果
            results = {
                'accuracies': {count: {'gibbs': [], 'em': []} for count in sequence_counts},
                'times': {count: {'gibbs': [], 'em': []} for count in sequence_counts},
                'evaluations': {count: {'gibbs': [], 'em': []} for count in sequence_counts}
            }
            
            for count in sequence_counts:
                for i in range(test_repeats):
                    # 生成测试数据
                    sequences, true_positions, _ = generate_test_sequences(count, length, motif_length)
                    
                    # Gibbs采样
                    start_time = time.time()
                    gibbs = GibbsSampler(sequences, motif_length, pseudocount)
                    positions, _ = gibbs.run(max_iterations=max_iterations, n_starts=n_starts)
                    gibbs_time = time.time() - start_time
                    
                    gibbs_eval = evaluate_motif_predictions(
                        positions, true_positions, sequences, motif_length
                    )
                    results['accuracies'][count]['gibbs'].append(gibbs_eval['average_score'])
                    results['times'][count]['gibbs'].append(gibbs_time)
                    results['evaluations'][count]['gibbs'].append(gibbs_eval)
                    
                    # EM算法
                    start_time = time.time()
                    em = ExpectationMaximization(sequences, motif_length, pseudocount)
                    positions, _ = em.find_motif(max_iterations=max_iterations)
                    em_time = time.time() - start_time
                    
                    em_eval = evaluate_motif_predictions(
                        positions, true_positions, sequences, motif_length
                    )
                    results['accuracies'][count]['em'].append(em_eval['average_score'])
                    results['times'][count]['em'].append(em_time)
                    results['evaluations'][count]['em'].append(em_eval)
                    
                    # 更新进度
                    current_step += 1
                    progress = (current_step / total_steps) * 100
                    progress_bar['value'] = progress
                    progress_label['text'] = f"比较进度: {progress:.1f}%"
                    progress_window.update()
            
            # 关闭进度窗口
            progress_window.destroy()
            
            # 显示比较结果
            self.display_algorithm_comparison(results, sequence_counts, max_iterations, n_starts)
            
        except Exception as e:
            self.result_text.delete("1.0", tk.END)
            self.result_text.insert(tk.END, f"分析错误: {str(e)}")

    def display_algorithm_comparison(self, results, sequence_counts, max_iterations, n_starts):
        """显示算法对比结果"""
        comparison_window = tk.Toplevel(self.root)
        comparison_window.title("Gibbs采样与EM算法性能对比")
        
        # 创建图表
        fig = Figure(figsize=(15, 10))
        canvas = FigureCanvasTkAgg(fig, master=comparison_window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 创建子图
        gs = fig.add_gridspec(2, 2)
        ax1 = fig.add_subplot(gs[0, 0])  # 准确率-序列数量关系
        ax2 = fig.add_subplot(gs[0, 1])  # 运行时间-序列数量关系
        ax3 = fig.add_subplot(gs[1, :])  # 箱线图比较
        
        # 1. 准确率-序列数量关系
        gibbs_acc_means = [np.mean(results['accuracies'][count]['gibbs']) for count in sequence_counts]
        em_acc_means = [np.mean(results['accuracies'][count]['em']) for count in sequence_counts]
        gibbs_acc_stds = [np.std(results['accuracies'][count]['gibbs']) for count in sequence_counts]
        em_acc_stds = [np.std(results['accuracies'][count]['em']) for count in sequence_counts]
        
        ax1.errorbar(sequence_counts, gibbs_acc_means, yerr=gibbs_acc_stds, 
                     fmt='o-', label='Gibbs样', capsize=5)
        ax1.errorbar(sequence_counts, em_acc_means, yerr=em_acc_stds, 
                     fmt='s-', label='EM算法', capsize=5)
        ax1.set_title('准确率与序列数量的关系', fontproperties='SimHei', fontsize=12)
        ax1.set_xlabel('序列数量', fontproperties='SimHei', fontsize=10)
        ax1.set_ylabel('准确率', fontproperties='SimHei', fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax1.legend(prop={'family': 'SimHei', 'size': 9})
        
        # 2. 运行时间-序列数量关系
        gibbs_time_means = [np.mean(results['times'][count]['gibbs']) for count in sequence_counts]
        em_time_means = [np.mean(results['times'][count]['em']) for count in sequence_counts]
        gibbs_time_stds = [np.std(results['times'][count]['gibbs']) for count in sequence_counts]
        em_time_stds = [np.std(results['times'][count]['em']) for count in sequence_counts]
        
        ax2.errorbar(sequence_counts, gibbs_time_means, yerr=gibbs_time_stds, 
                     fmt='o-', label='Gibbs采样', capsize=5)
        ax2.errorbar(sequence_counts, em_time_means, yerr=em_time_stds, 
                     fmt='s-', label='EM算法', capsize=5)
        ax2.set_title('运行时间与序列数量的关系', fontproperties='SimHei', fontsize=12)
        ax2.set_xlabel('序列数量', fontproperties='SimHei', fontsize=10)
        ax2.set_ylabel('运行时间 (秒)', fontproperties='SimHei', fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.3)
        ax2.legend(prop={'family': 'SimHei', 'size': 9})
        
        # 3. 箱线图比较
        all_gibbs_acc = []
        all_em_acc = []
        all_gibbs_time = []
        all_em_time = []
        
        for count in sequence_counts:
            all_gibbs_acc.extend(results['accuracies'][count]['gibbs'])
            all_em_acc.extend(results['accuracies'][count]['em'])
            all_gibbs_time.extend(results['times'][count]['gibbs'])
            all_em_time.extend(results['times'][count]['em'])
        
        data = [all_gibbs_acc, all_em_acc]
        labels = ['Gibbs采样', 'EM算法']
        
        bp = ax3.boxplot(data, labels=labels, patch_artist=True)
        colors = ['lightblue', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax3.set_title('算法整体准确率分布对比', fontproperties='SimHei', fontsize=12)
        ax3.set_ylabel('准确率', fontproperties='SimHei', fontsize=10)
        ax3.grid(True, linestyle='--', alpha=0.3)
        
        fig.tight_layout(pad=1.0)
        
        # 添加文本结果
        text_frame = ttk.Frame(comparison_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        result_text = scrolledtext.ScrolledText(text_frame, width=60, height=10)
        result_text.pack(fill=tk.BOTH, expand=True)
        
        # 显示统计信息
        result_text.insert(tk.END, "算法性能对比结果:\n\n")
        result_text.insert(tk.END, f"测试参数:\n")
        result_text.insert(tk.END, f"  序列数量范围: {min(sequence_counts)}-{max(sequence_counts)}\n")
        result_text.insert(tk.END, f"  迭代次数: {max_iterations}\n")
        result_text.insert(tk.END, f"  重启次数: {n_starts}\n\n")
        
        # 显示每个序列数量的结果
        for count in sequence_counts:
            result_text.insert(tk.END, f"\n序列数量 {count}:\n")
            
            # Gibbs采样结果
            result_text.insert(tk.END, f"  Gibbs采样:\n")
            gibbs_evals = results['evaluations'][count]['gibbs']
            gibbs_avg = np.mean([eval_result['average_score'] for eval_result in gibbs_evals])
            gibbs_perfect = np.mean([eval_result['perfect_ratio'] for eval_result in gibbs_evals])
            gibbs_time = np.mean(results['times'][count]['gibbs'])
            
            result_text.insert(tk.END, f"    平均碱基匹配率: {gibbs_avg:.2%}\n")
            result_text.insert(tk.END, f"    完全匹配比例: {gibbs_perfect:.2%}\n")
            result_text.insert(tk.END, f"    平均运行时间: {gibbs_time:.2f}秒\n")
            
            # 添加详细的分布信息
            all_scores = [score for eval_result in gibbs_evals 
                         for score in eval_result['sequence_scores']]
            score_distribution = {
                '100%': sum(1 for s in all_scores if s == 1.0),
                '>=75%': sum(1 for s in all_scores if s >= 0.75),
                '>=50%': sum(1 for s in all_scores if s >= 0.50),
                '<50%': sum(1 for s in all_scores if s < 0.50)
            }
            total_seqs = len(all_scores)
            
            result_text.insert(tk.END, f"    匹配率分布:\n")
            for threshold, count in score_distribution.items():
                result_text.insert(tk.END, f"      {threshold}: {count/total_seqs:.2%}\n")
            
            # EM算法结果
            result_text.insert(tk.END, f"  EM算法:\n")
            em_evals = results['evaluations'][count]['em']
            em_avg = np.mean([eval_result['average_score'] for eval_result in em_evals])
            em_perfect = np.mean([eval_result['perfect_ratio'] for eval_result in em_evals])
            em_time = np.mean(results['times'][count]['em'])
            
            result_text.insert(tk.END, f"    平均碱基匹配率: {em_avg:.2%}\n")
            result_text.insert(tk.END, f"    完全匹配比例: {em_perfect:.2%}\n")
            result_text.insert(tk.END, f"    平均运行时间: {em_time:.2f}秒\n")
            
            # 添加详细的分布信息
            all_scores = [score for eval_result in em_evals 
                         for score in eval_result['sequence_scores']]
            score_distribution = {
                '100%': sum(1 for s in all_scores if s == 1.0),
                '>=75%': sum(1 for s in all_scores if s >= 0.75),
                '>=50%': sum(1 for s in all_scores if s >= 0.50),
                '<50%': sum(1 for s in all_scores if s < 0.50)
            }
            total_seqs = len(all_scores)
            
            result_text.insert(tk.END, f"    匹配率分布:\n")
            for threshold, count in score_distribution.items():
                result_text.insert(tk.END, f"      {threshold}: {count/total_seqs:.2%}\n")

    def validate_inputs(self):
        """验证输入参数"""
        try:
            motif_length = int(self.motif_length.get())
            seq_length = int(self.seq_length.get())
            seq_count = int(self.seq_count.get())
            pseudocount = float(self.pseudocount.get())
            
            if motif_length < 4:
                raise ValueError("模体长度必须大于等于4")
            if seq_length < motif_length * 2:
                raise ValueError(f"序列长度必须至少是模体长度的2倍 (>= {motif_length * 2})")
            if seq_count < 2:
                raise ValueError("序列数量必须大于等于2")
            if pseudocount <= 0:
                raise ValueError("伪计数必须大于0")
                
            return True
            
        except ValueError as e:
            self.result_text.delete("1.0", tk.END)
            self.result_text.insert(tk.END, f"参数错误: {str(e)}")
            return False




