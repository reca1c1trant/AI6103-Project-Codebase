"""
评估LoRA微调后的BERT模型

用法:
python evaluate_lora.py \
    --test_path test.csv \
    --model_path ./outputs_lora/best_model \
    --output_dir ./evaluation_results_lora
"""

import os
import torch
import pandas as pd
import argparse
from tqdm import tqdm
from collections import Counter
import string
import re
import json
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    set_seed
)
from peft import PeftModel

class LoRAEvaluator:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f" 评估设置")
        print(f"   - Device: {self.device}")
        print(f"   - Test Data: {args.test_path}")
        print(f"   - LoRA Model: {args.model_path}")
        
        self.load_test_data()
        self.load_model()
    
    def load_test_data(self):
        """加载测试数据"""
        print(f"\n 加载测试数据...")
        df = pd.read_csv(self.args.test_path)
        df = df.dropna(subset=['context', 'question', 'answer'])
        
        self.test_data = []
        for idx, row in df.iterrows():
            self.test_data.append({
                'id': str(idx),
                'context': str(row['context']).strip(),
                'question': str(row['question']).strip(),
                'answer': str(row['answer']).strip()
            })
        
        print(f"    加载 {len(self.test_data):,} 个测试样本")
    
    def load_model(self):
        """加载LoRA模型"""
        print(f"\n 加载LoRA模型...")
        
        # 检查是否是LoRA模型
        adapter_config_path = os.path.join(self.args.model_path, "adapter_config.json")
        
        if os.path.exists(adapter_config_path):
            # 加载LoRA模型
            print(f"   - 检测到LoRA适配器")
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
                base_model_name = adapter_config.get('base_model_name_or_path', '')
                print(f"   - 基础模型: {base_model_name}")
            
            # 先加载基础模型
            base_model = AutoModelForQuestionAnswering.from_pretrained(base_model_name)
            # 加载LoRA适配器
            self.model = PeftModel.from_pretrained(base_model, self.args.model_path)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path)
            
            print(f"    LoRA模型加载完成")
        else:
            # 普通模型
            print(f"   - 加载普通模型")
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path)
            self.model = AutoModelForQuestionAnswering.from_pretrained(
                self.args.model_path
            ).to(self.device)
            self.model.eval()
            print(f"    模型加载完成")
        
        print(f"   - 模型层数: {self.model.config.num_hidden_layers}")
    
    def normalize_answer(self, s):
        """标准化答案"""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        
        def white_space_fix(text):
            return ' '.join(text.split())
        
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        
        def lower(text):
            return text.lower()
        
        return white_space_fix(remove_articles(remove_punc(lower(s))))
    
    def compute_exact_match(self, prediction, ground_truth):
        """计算EM"""
        return int(self.normalize_answer(prediction) == self.normalize_answer(ground_truth))
    
    def compute_f1(self, prediction, ground_truth):
        """计算F1"""
        pred_tokens = self.normalize_answer(prediction).split()
        truth_tokens = self.normalize_answer(ground_truth).split()
        
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return int(pred_tokens == truth_tokens)
        
        common_tokens = Counter(pred_tokens) & Counter(truth_tokens)
        num_common = sum(common_tokens.values())
        
        if num_common == 0:
            return 0
        
        precision = num_common / len(pred_tokens)
        recall = num_common / len(truth_tokens)
        f1 = 2 * (precision * recall) / (precision + recall)
        
        return f1
    
    def evaluate(self):
        """评估模型"""
        print(f"\n{'='*70}")
        print(f" 开始评估")
        print(f"{'='*70}")
        
        total_em = 0
        total_f1 = 0
        total_start_correct = 0
        total_end_correct = 0
        num_samples = len(self.test_data)
        
        results = []
        
        for sample in tqdm(self.test_data, desc="Evaluating"):
            # Tokenize
            inputs = self.tokenizer(
                sample['question'],
                sample['context'],
                max_length=self.args.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt',
                return_offsets_mapping=True
            )
            
            offset_mapping = inputs.pop('offset_mapping')[0]
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 计算真实的start/end位置
            answer_start_char = sample['context'].find(sample['answer'])
            answer_end_char = answer_start_char + len(sample['answer'])
            
            true_start_pos = 0
            true_end_pos = 0
            for idx, (start, end) in enumerate(offset_mapping.tolist()):
                if start <= answer_start_char < end:
                    true_start_pos = idx
                if start < answer_end_char <= end:
                    true_end_pos = idx
                    break
            
            # 模型预测
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            
            pred_start_pos = torch.argmax(start_logits, dim=1).item()
            pred_end_pos = torch.argmax(end_logits, dim=1).item()
            
            if pred_end_pos < pred_start_pos:
                pred_end_pos = pred_start_pos
            
            # Start/End Accuracy
            if pred_start_pos == true_start_pos:
                total_start_correct += 1
            if pred_end_pos == true_end_pos:
                total_end_correct += 1
            
            # 解码答案
            input_ids = inputs['input_ids'][0]
            pred_tokens = input_ids[pred_start_pos:pred_end_pos+1]
            predicted_answer = self.tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()
            
            # EM和F1
            em = self.compute_exact_match(predicted_answer, sample['answer'])
            f1 = self.compute_f1(predicted_answer, sample['answer'])
            
            total_em += em
            total_f1 += f1
            
            results.append({
                'id': sample['id'],
                'question': sample['question'],
                'context': sample['context'][:100] + '...',
                'ground_truth': sample['answer'],
                'prediction': predicted_answer,
                'em': em,
                'f1': f1,
                'start_correct': int(pred_start_pos == true_start_pos),
                'end_correct': int(pred_end_pos == true_end_pos)
            })
        
        avg_em = (total_em / num_samples) * 100
        avg_f1 = (total_f1 / num_samples) * 100
        start_accuracy = (total_start_correct / num_samples) * 100
        end_accuracy = (total_end_correct / num_samples) * 100
        
        # 打印结果
        self.print_results(num_samples, avg_em, avg_f1, start_accuracy, end_accuracy, results)
        
        # 保存结果
        if self.args.output_dir:
            self.save_results(num_samples, avg_em, avg_f1, start_accuracy, end_accuracy, results)
    
    def print_results(self, num_samples, avg_em, avg_f1, start_accuracy, end_accuracy, results):
        """打印结果"""
        print(f"\n{'='*70}")
        print(f" 评估结果")
        print(f"{'='*70}")
        
        print(f"\nNumber of samples: {num_samples}")
        print(f"\nExact Match (EM):  {avg_em:.2f}%")
        print(f"F1 Score:          {avg_f1:.2f}%")
        print(f"Start Accuracy:    {start_accuracy:.2f}%")
        print(f"End Accuracy:      {end_accuracy:.2f}%")
        
        # 样例
        print(f"\n{'='*70}")
        print(f" 样例预测 (前5个)")
        print(f"{'='*70}")
        
        for i in range(min(5, len(results))):
            sample = results[i]
            
            print(f"\n[样例 {i+1}]")
            print(f"  Question:     {sample['question']}")
            print(f"  Ground Truth: {sample['ground_truth']}")
            print(f"  Prediction:   {sample['prediction']} (EM:{sample['em']}, F1:{sample['f1']:.2f})")
    
    def save_results(self, num_samples, avg_em, avg_f1, start_accuracy, end_accuracy, results):
        """保存结果"""
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        summary = {
            'num_samples': num_samples,
            'model_path': self.args.model_path,
            'exact_match': avg_em,
            'f1_score': avg_f1,
            'start_accuracy': start_accuracy,
            'end_accuracy': end_accuracy
        }
        
        summary_path = os.path.join(self.args.output_dir, 'evaluation_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n💾 汇总结果: {summary_path}")
        
        # 详细结果
        detailed_results = []
        for sample in results:
            detailed_results.append({
                'id': sample['id'],
                'question': sample['question'],
                'ground_truth': sample['ground_truth'],
                'prediction': sample['prediction'],
                'em': sample['em'],
                'f1': sample['f1'],
                'start_correct': sample['start_correct'],
                'end_correct': sample['end_correct']
            })
        
        detailed_df = pd.DataFrame(detailed_results)
        detailed_path = os.path.join(self.args.output_dir, 'detailed_results.csv')
        detailed_df.to_csv(detailed_path, index=False)
        print(f" 详细结果: {detailed_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate LoRA Fine-tuned BERT on SQuAD')
    
    parser.add_argument('--test_path', type=str, default='test.csv',
                       help='测试数据CSV路径')
    parser.add_argument('--model_path', type=str, 
                       default='./outputs_lora/best_model',
                       help='LoRA模型路径')
    parser.add_argument('--max_length', type=int, default=384,
                       help='最大序列长度')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results_lora',
                       help='结果输出目录')
    
    args = parser.parse_args()
    
    set_seed(42)
    
    evaluator = LoRAEvaluator(args)
    evaluator.evaluate()


if __name__ == '__main__':
    main()