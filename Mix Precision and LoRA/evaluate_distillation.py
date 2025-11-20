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

class DistillationEvaluator:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f" 评估设置")
        print(f"   - Device: {self.device}")
        print(f"   - Test Data: {args.test_path}")
        print(f"   - Student Distilled Model: {args.student_distilled}")
        
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
        """加载蒸馏后的Student模型"""
        print(f"\n 加载模型...")
        
        print(f"   - 加载Student Distilled: {self.args.student_distilled}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.student_distilled)
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            self.args.student_distilled
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
    
    def predict_answer(self, model, tokenizer, question, context):
        """预测答案"""
        inputs = tokenizer(
            question,
            context,
            max_length=self.args.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        
        start_idx = torch.argmax(start_logits, dim=1).item()
        end_idx = torch.argmax(end_logits, dim=1).item()
        
        if end_idx < start_idx:
            end_idx = start_idx
        
        input_ids = inputs['input_ids'][0]
        answer_tokens = input_ids[start_idx:end_idx+1]
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
        
        return answer.strip()
    
    def evaluate_model(self, model, tokenizer, model_name):
        """评估模型"""
        print(f"\n{'='*70}")
        print(f" 评估 {model_name}")
        print(f"{'='*70}")
        
        total_em = 0
        total_f1 = 0
        total_start_correct = 0
        total_end_correct = 0
        num_samples = len(self.test_data)
        
        results = []
        
        for sample in tqdm(self.test_data, desc=f"Evaluating {model_name}"):
            # Tokenize
            inputs = tokenizer(
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
                outputs = model(**inputs)
            
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
            predicted_answer = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()
            
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
        
        return {
            'model_name': model_name,
            'num_samples': num_samples,
            'exact_match': avg_em,
            'f1_score': avg_f1,
            'start_accuracy': start_accuracy,
            'end_accuracy': end_accuracy,
            'detailed_results': results
        }
    
    def run_evaluation(self):
        """运行评估"""
        print(f"\n{'='*70}")
        print(f" 开始评估")
        print(f"{'='*70}")
        
        # 评估蒸馏后模型
        results = self.evaluate_model(
            self.model,
            self.tokenizer,
            "Student Distilled"
        )
        
        # 打印和保存结果
        self.print_results(results)
        
        if self.args.output_dir:
            self.save_results(results)
    
    def print_results(self, results):
        """打印结果"""
        print(f"\n{'='*70}")
        print(f" 评估结果")
        print(f"{'='*70}")
        
        print(f"\nNumber of samples: {results['num_samples']}")
        print(f"\nExact Match (EM):  {results['exact_match']:.2f}%")
        print(f"F1 Score:          {results['f1_score']:.2f}%")
        print(f"Start Accuracy:    {results['start_accuracy']:.2f}%")
        print(f"End Accuracy:      {results['end_accuracy']:.2f}%")
        
        # 样例
        print(f"\n{'='*70}")
        print(f" 样例预测 (前5个)")
        print(f"{'='*70}")
        
        for i in range(min(5, len(results['detailed_results']))):
            sample = results['detailed_results'][i]
            
            print(f"\n[样例 {i+1}]")
            print(f"  Question:     {sample['question']}")
            print(f"  Ground Truth: {sample['ground_truth']}")
            print(f"  Prediction:   {sample['prediction']} (EM:{sample['em']}, F1:{sample['f1']:.2f})")
    
    def save_results(self, results):
        """保存结果"""
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        summary = {
            'num_samples': results['num_samples'],
            'model_name': self.args.student_distilled,
            'exact_match': results['exact_match'],
            'f1_score': results['f1_score'],
            'start_accuracy': results['start_accuracy'],
            'end_accuracy': results['end_accuracy']
        }
        
        summary_path = os.path.join(self.args.output_dir, 'evaluation_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n 汇总结果: {summary_path}")
        
        # 详细结果
        detailed_results = []
        for sample in results['detailed_results']:
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
    parser = argparse.ArgumentParser(description='Evaluate Distilled BERT on SQuAD')
    
    parser.add_argument('--test_path', type=str, default='test.csv',
                       help='测试数据CSV路径')
    parser.add_argument('--student_distilled', type=str, 
                       default='./outputs_student_distilled_fp32/best_student_model',
                       help='蒸馏后的Student模型路径')
    parser.add_argument('--max_length', type=int, default=384,
                       help='最大序列长度')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='结果输出目录')
    
    args = parser.parse_args()
    
    set_seed(42)
    
    evaluator = DistillationEvaluator(args)
    evaluator.run_evaluation()


if __name__ == '__main__':
    main()

"""
使用示例:

python evaluate_distillation.py \
    --test_path test.csv \
    --student_distilled ./outputs_fp32/best_student_model \
    --output_dir ./evaluation_results_fp32
"""