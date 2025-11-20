"""
Step 2: Knowledge Distillation - Train Student BERT-12 with Teacher BERT-12

用法:
torchrun --nproc_per_node=8 train_student_distillation.py     --teacher_path google-bert/bert-large-uncased-whole-word-masking-finetuned-squad     --student_model bert-base-uncased     --train_path SQuAD-v1.1.csv     --num_epochs 6     --batch_size 16     --learning_rate 3e-5     --alpha 0.5     --precision bf16     --output_dir ./outputs_student_distilled_bf16

"""

import os
import time
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    get_cosine_schedule_with_warmup,
    set_seed
)
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import argparse
import pandas as pd
import json
import glob
import shutil

class SQuADQADataset(Dataset):
    """SQuAD QA数据集"""
    def __init__(self, encodings):
        self.encodings = encodings
    
    def __len__(self):
        return len(self.encodings['input_ids'])
    
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

class DistillationTrainer:
    def __init__(self, args):
        self.args = args
        self.setup_distributed()
        self.setup_model_and_data()
        self.setup_training()
        
    def setup_distributed(self):
        """设置分布式训练"""
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.local_rank = int(os.environ['LOCAL_RANK'])
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
            
        torch.cuda.set_device(self.local_rank)
        
        if self.world_size > 1:
            dist.init_process_group(backend='nccl')
        
        self.is_main = self.rank == 0
        
        if self.is_main:
            print(f"\n{'='*70}")
            print(f" STEP 2: Knowledge Distillation (12-layer → 12-layer)")
            print(f"{'='*70}")
            print(f" 分布式训练设置:")
            print(f"   - World Size: {self.world_size}")
            print(f"   - Rank: {self.rank}")
            print(f"   - Local Rank: {self.local_rank}")
    
    def load_squad_qa_data(self, csv_path):
        """从CSV加载SQuAD QA数据"""
        if self.is_main:
            print(f"\n 加载数据: {csv_path}")
        
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=['context', 'question', 'answer'])
        
        if self.is_main:
            print(f"   - 有效样本: {len(df):,}")
        
        contexts = df['context'].tolist()
        questions = df['question'].tolist()
        answers = df['answer'].tolist()
        
        return contexts, questions, answers
    
    def prepare_qa_features(self, contexts, questions, answers, tokenizer):
        """准备QA特征"""
        encodings = {
            'input_ids': [],
            'attention_mask': [],
            'start_positions': [],
            'end_positions': []
        }
        
        if self.is_main:
            print(f"\n Tokenization...")
        
        for context, question, answer in tqdm(
            zip(contexts, questions, answers),
            total=len(contexts),
            disable=not self.is_main,
            desc="Encoding"
        ):
            encoding = tokenizer(
                question,
                context,
                max_length=self.args.max_length,
                truncation=True,
                padding='max_length',
                return_offsets_mapping=True,
                return_tensors='pt'
            )
            
            answer_start = context.find(answer)
            if answer_start == -1:
                continue
            
            answer_end = answer_start + len(answer)
            offset_mapping = encoding['offset_mapping'][0].tolist()
            
            start_position = 0
            end_position = 0
            
            for idx, (start, end) in enumerate(offset_mapping):
                if start <= answer_start < end:
                    start_position = idx
                if start < answer_end <= end:
                    end_position = idx
                    break
            
            encodings['input_ids'].append(encoding['input_ids'][0].tolist())
            encodings['attention_mask'].append(encoding['attention_mask'][0].tolist())
            encodings['start_positions'].append(start_position)
            encodings['end_positions'].append(end_position)
        
        if self.is_main:
            print(f"    编码完成: {len(encodings['input_ids']):,} 样本")
        
        return encodings
    
    def setup_model_and_data(self):
        """加载Teacher和Student模型"""
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.teacher_path)
        
        # 加载训练数据
        train_contexts, train_questions, train_answers = self.load_squad_qa_data(
            self.args.train_path
        )
        
        train_encodings = self.prepare_qa_features(
            train_contexts, train_questions, train_answers, self.tokenizer
        )
        self.train_dataset = SQuADQADataset(train_encodings)
        
        # 验证集
        if self.args.eval_path:
            eval_contexts, eval_questions, eval_answers = self.load_squad_qa_data(
                self.args.eval_path
            )
            eval_encodings = self.prepare_qa_features(
                eval_contexts, eval_questions, eval_answers, self.tokenizer
            )
            self.eval_dataset = SQuADQADataset(eval_encodings)
        else:
            train_size = int(len(train_encodings['input_ids']) * 0.9)
            train_enc = {k: v[:train_size] for k, v in train_encodings.items()}
            eval_enc = {k: v[train_size:] for k, v in train_encodings.items()}
            
            self.train_dataset = SQuADQADataset(train_enc)
            self.eval_dataset = SQuADQADataset(eval_enc)
            
            if self.is_main:
                print(f"   - 训练集: {len(self.train_dataset):,}")
                print(f"   - 验证集: {len(self.eval_dataset):,}")
        
        # 加载Teacher模型（冻结）
        if self.is_main:
            print(f"\n 加载Teacher模型: {self.args.teacher_path}")
        
        self.teacher = AutoModelForQuestionAnswering.from_pretrained(
            self.args.teacher_path
        ).to(self.local_rank)
        self.teacher.eval()  # 设为eval模式
        
        # 冻结Teacher参数
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        if self.is_main:
            print(f"   - 模型类型: BERT-base (Teacher)")
            print(f"   - 层数: {self.teacher.config.num_hidden_layers}")
            print(f"   - 状态: 冻结 (用于生成soft labels)")
        
        # 加载Student模型（训练）
        if self.is_main:
            print(f"\n 加载Student模型: {self.args.student_model}")
        
        self.student = AutoModelForQuestionAnswering.from_pretrained(
            self.args.student_model
        ).to(self.local_rank)
        
        if self.is_main:
            print(f"   - 模型类型: BERT-base (Student)")
            print(f"   - 层数: {self.student.config.num_hidden_layers}")
            print(f"   - 状态: 可训练")
        
        # DDP包装（只包装Student）
        if self.world_size > 1:
            self.student = DDP(
                self.student,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False
            )
        
        if self.is_main:
            student_params = sum(p.numel() for p in self.student.parameters())
            trainable_params = sum(p.numel() for p in self.student.parameters() if p.requires_grad)
            print(f"   - Student参数量: {student_params:,}")
            print(f"   - 可训练参数: {trainable_params:,}")
    
    def setup_training(self):
        """设置训练组件"""
        # DataLoader
        if self.world_size > 1:
            train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                seed=42
            )
            eval_sampler = DistributedSampler(
                self.eval_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False
            )
        else:
            train_sampler = None
            eval_sampler = None
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=4,
            pin_memory=True
        )
        
        self.eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.args.batch_size * 2,
            sampler=eval_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        # 优化器（只优化Student）
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.student.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.args.weight_decay,
            },
            {
                'params': [p for n, p in self.student.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            }
        ]
        
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 学习率调度器
        num_training_steps = len(self.train_loader) * self.args.num_epochs
        num_warmup_steps = int(num_training_steps * self.args.warmup_ratio)
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Mixed Precision
        self.setup_mixed_precision()
        
        if self.is_main:
            print(f"\n  知识蒸馏配置:")
            print(f"   - Temperature: {self.args.temperature}")
            print(f"   - Alpha (hard loss weight): {self.args.alpha}")
            print(f"   - 1-Alpha (distillation loss weight): {1-self.args.alpha}")
            print(f"   - Precision: {self.args.precision}")
            print(f"   - Batch Size (per device): {self.args.batch_size}")
            print(f"   - Global Batch Size: {self.args.batch_size * self.world_size}")
            print(f"   - 学习率: {self.args.learning_rate}")
            print(f"   - 学习率调度: Cosine Annealing with warmup")
            print(f"   - Warmup步数: {num_warmup_steps:,} ({self.args.warmup_ratio*100}%)")
            print(f"   - 总训练步数: {num_training_steps:,}")
    
    def setup_mixed_precision(self):
        """设置Mixed Precision"""
        self.use_amp = False
        self.scaler = None
        self.amp_dtype = torch.float32
        
        if self.args.precision == "fp16":
            self.use_amp = True
            self.amp_dtype = torch.float16
            self.scaler = GradScaler()
            if self.is_main:
                print(f"    启用 FP16 训练")
        elif self.args.precision == "bf16":
            if torch.cuda.is_bf16_supported():
                self.use_amp = True
                self.amp_dtype = torch.bfloat16
                if self.is_main:
                    print(f"    启用 BF16 训练")
            else:
                if self.is_main:
                    print(f"     GPU不支持BF16，回退到FP32")
        elif self.args.precision == "fp32":
            if self.is_main:
                print(f"    使用 FP32 训练")
    
    def distillation_loss(self, student_logits, teacher_logits, temperature):
        """
        计算蒸馏损失 (KL散度)
        
        Args:
            student_logits: Student模型的logits (start_logits, end_logits)
            teacher_logits: Teacher模型的logits (start_logits, end_logits)
            temperature: 温度参数
        """
        student_start, student_end = student_logits
        teacher_start, teacher_end = teacher_logits
        
        # 对start position的KL散度
        start_loss = F.kl_div(
            F.log_softmax(student_start / temperature, dim=-1),
            F.softmax(teacher_start / temperature, dim=-1),
            reduction='batchmean'
        ) * (temperature ** 2)
        
        # 对end position的KL散度
        end_loss = F.kl_div(
            F.log_softmax(student_end / temperature, dim=-1),
            F.softmax(teacher_end / temperature, dim=-1),
            reduction='batchmean'
        ) * (temperature ** 2)
        
        return (start_loss + end_loss) / 2
    
    def train_epoch(self, epoch):
        """训练一个epoch（带知识蒸馏）"""
        self.student.train()
        self.teacher.eval()  # Teacher始终保持eval
        
        total_loss = 0
        total_hard_loss = 0
        total_distill_loss = 0
        
        if self.world_size > 1:
            self.train_loader.sampler.set_epoch(epoch)
        
        pbar = tqdm(self.train_loader, disable=not self.is_main, 
                   desc=f"Epoch {epoch+1}/{self.args.num_epochs}")
        
        for batch in pbar:
            batch = {k: v.to(self.local_rank) for k, v in batch.items()}
            
            # Forward pass
            if self.use_amp:
                with autocast(device_type='cuda', dtype=self.amp_dtype, enabled=True):
                    # Student forward
                    student_outputs = self.student(**batch)
                    hard_loss = student_outputs.loss
                    
                    # Teacher forward（无梯度）
                    with torch.no_grad():
                        teacher_outputs = self.teacher(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask']
                        )
                    
                    # 计算蒸馏损失
                    distill_loss = self.distillation_loss(
                        (student_outputs.start_logits, student_outputs.end_logits),
                        (teacher_outputs.start_logits, teacher_outputs.end_logits),
                        self.args.temperature
                    )
                    
                    # 组合损失
                    loss = self.args.alpha * hard_loss + (1 - self.args.alpha) * distill_loss
            else:
                # Student forward
                student_outputs = self.student(**batch)
                hard_loss = student_outputs.loss
                
                # Teacher forward（无梯度）
                with torch.no_grad():
                    teacher_outputs = self.teacher(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask']
                    )
                
                # 计算蒸馏损失
                distill_loss = self.distillation_loss(
                    (student_outputs.start_logits, student_outputs.end_logits),
                    (teacher_outputs.start_logits, teacher_outputs.end_logits),
                    self.args.temperature
                )
                
                # 组合损失
                loss = self.args.alpha * hard_loss + (1 - self.args.alpha) * distill_loss
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), 
                                              self.args.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), 
                                              self.args.max_grad_norm)
                self.optimizer.step()
            
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            total_loss += loss.item()
            total_hard_loss += hard_loss.item()
            total_distill_loss += distill_loss.item()
            
            if self.is_main:
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'hard': f"{hard_loss.item():.4f}",
                    'distill': f"{distill_loss.item():.4f}",
                    'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
                })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_hard_loss = total_hard_loss / len(self.train_loader)
        avg_distill_loss = total_distill_loss / len(self.train_loader)
        
        return avg_loss, avg_hard_loss, avg_distill_loss
    
    def evaluate(self):
        """评估Student模型"""
        self.student.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_loader, disable=not self.is_main, desc="Eval"):
                batch = {k: v.to(self.local_rank) for k, v in batch.items()}
                
                if self.use_amp:
                    with autocast(device_type='cuda', dtype=self.amp_dtype, enabled=True):
                        outputs = self.student(**batch)
                        loss = outputs.loss
                else:
                    outputs = self.student(**batch)
                    loss = outputs.loss
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.eval_loader)
        
        # 多卡同步
        if self.world_size > 1:
            avg_loss_tensor = torch.tensor(avg_loss).to(self.local_rank)
            dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss_tensor.item() / self.world_size
        
        return avg_loss
    
    def delete_old_checkpoints(self):
        """删除旧checkpoint"""
        if not self.is_main:
            return
        
        checkpoint_pattern = os.path.join(self.args.output_dir, 'checkpoint-epoch-*')
        checkpoints = sorted(glob.glob(checkpoint_pattern))
        
        if len(checkpoints) > 1:
            for old_checkpoint in checkpoints[:-1]:
                if self.is_main:
                    print(f"   🗑️  删除旧checkpoint: {old_checkpoint}")
                shutil.rmtree(old_checkpoint)
    
    def save_checkpoint(self, epoch, eval_loss, is_best=False):
        """保存checkpoint"""
        if not self.is_main:
            return
        
        self.delete_old_checkpoints()
        
        # 获取真实的student模型（去除DDP wrapper）
        student_model = self.student.module if self.world_size > 1 else self.student
        
        # 保存最新checkpoint
        checkpoint_dir = os.path.join(self.args.output_dir, 
                                     f"checkpoint-epoch-{epoch+1}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        student_model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # 保存训练状态
        checkpoint_state = {
            'epoch': epoch,
            'eval_loss': eval_loss,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        
        if self.scaler is not None:
            checkpoint_state['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint_state, 
                  os.path.join(checkpoint_dir, 'training_state.pt'))
        
        print(f"    保存checkpoint: {checkpoint_dir}")
        
        # 保存best student model
        if is_best:
            best_model_dir = os.path.join(self.args.output_dir, "best_student_model")
            os.makedirs(best_model_dir, exist_ok=True)
            
            student_model.save_pretrained(best_model_dir)
            self.tokenizer.save_pretrained(best_model_dir)
            
            best_info = {
                'epoch': epoch + 1,
                'eval_loss': eval_loss,
                'precision': self.args.precision,
                'model_type': 'student',
                'num_layers': 12,
                'teacher_path': self.args.teacher_path,
                'temperature': self.args.temperature,
                'alpha': self.args.alpha,
            }
            
            with open(os.path.join(best_model_dir, 'student_info.json'), 'w') as f:
                json.dump(best_info, f, indent=2)
            
            print(f"    保存最佳Student模型: {best_model_dir}")
    
    def train(self):
        """完整训练流程"""
        start_time = time.time()
        best_eval_loss = float('inf')
        
        training_stats = {
            'model_type': 'student_distilled',
            'teacher_path': self.args.teacher_path,
            'student_model': self.args.student_model,
            'temperature': self.args.temperature,
            'alpha': self.args.alpha,
            'precision': self.args.precision,
            'batch_size': self.args.batch_size,
            'world_size': self.world_size,
            'learning_rate': self.args.learning_rate,
            'epochs': [],
        }
        
        for epoch in range(self.args.num_epochs):
            if self.is_main:
                print(f"\n{'='*70}")
                print(f" Epoch {epoch+1}/{self.args.num_epochs}")
                print(f"{'='*70}")
            
            epoch_start = time.time()
            
            # 训练（带蒸馏）
            train_loss, hard_loss, distill_loss = self.train_epoch(epoch)
            train_time = time.time() - epoch_start
            
            # 评估
            eval_start = time.time()
            eval_loss = self.evaluate()
            eval_time = time.time() - eval_start
            
            # 判断最佳
            is_best = eval_loss < best_eval_loss
            if is_best:
                best_eval_loss = eval_loss
            
            # 记录统计
            epoch_stats = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'hard_loss': hard_loss,
                'distill_loss': distill_loss,
                'eval_loss': eval_loss,
                'train_time_minutes': train_time / 60,
                'eval_time_minutes': eval_time / 60,
                'is_best': is_best,
                'learning_rate': self.scheduler.get_last_lr()[0],
            }
            training_stats['epochs'].append(epoch_stats)
            
            if self.is_main:
                print(f"\n Epoch {epoch+1} 结果:")
                print(f"   - Total Loss:  {train_loss:.4f}")
                print(f"   - Hard Loss:   {hard_loss:.4f} (α={self.args.alpha})")
                print(f"   - Distill Loss: {distill_loss:.4f} (1-α={1-self.args.alpha})")
                print(f"   - Eval Loss:   {eval_loss:.4f} {'🌟 (BEST)' if is_best else ''}")
                print(f"   - Train Time:  {train_time/60:.2f} min")
                print(f"   - Eval Time:   {eval_time/60:.2f} min")
                print(f"   - Learning Rate: {self.scheduler.get_last_lr()[0]:.2e}")
            
            # 保存checkpoint
            self.save_checkpoint(epoch, eval_loss, is_best=is_best)
        
        total_time = time.time() - start_time
        training_stats['total_time_hours'] = total_time / 3600
        training_stats['best_eval_loss'] = best_eval_loss
        
        if self.is_main:
            print(f"\n{'='*70}")
            print(f" 知识蒸馏训练完成!")
            print(f"{'='*70}")
            print(f"   - 总耗时: {total_time/3600:.2f} 小时")
            print(f"   - 最佳Eval Loss: {best_eval_loss:.4f}")
            print(f"   - Best Student模型路径: {os.path.join(self.args.output_dir, 'best_student_model')}")
            print(f"{'='*70}\n")
            
            stats_path = os.path.join(self.args.output_dir, 'distillation_stats.json')
            with open(stats_path, 'w') as f:
                json.dump(training_stats, f, indent=2)
            print(f" 蒸馏训练统计: {stats_path}\n")
        
        if self.world_size > 1:
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description='Step 2: Knowledge Distillation from Teacher to Student')
    
    # 数据参数
    parser.add_argument('--train_path', type=str, required=True,
                       help='训练数据CSV路径')
    parser.add_argument('--eval_path', type=str, default=None,
                       help='验证数据CSV路径（可选）')
    
    # 模型参数
    parser.add_argument('--teacher_path', type=str, required=True,
                       help='Teacher模型路径（Step 1训练好的best_teacher_model）')
    parser.add_argument('--student_model', type=str, default='bert-base-uncased',
                       help='Student初始模型（12层）')
    parser.add_argument('--max_length', type=int, default=512,
                       help='最大序列长度')
    
    # 蒸馏参数
    parser.add_argument('--temperature', type=float, default=6.0,
                       help='蒸馏温度（越大越soft）')
    parser.add_argument('--alpha', type=float, default=0.2,
                       help='Hard loss权重（1-alpha为distillation loss权重）')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='每张卡的batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-5,
                       help='学习率（通常比teacher训练时稍大）')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='权重衰减')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='梯度裁剪')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                       help='Warmup比例')
    
    # Mixed Precision
    parser.add_argument('--precision', type=str, default='bf16',
                       choices=['fp32', 'fp16', 'bf16'],
                       help='训练精度')
    
    # 输出
    parser.add_argument('--output_dir', type=str, default='./outputs_student_distilled',
                       help='输出目录')
    
    args = parser.parse_args()
    
    
    set_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)
    
    trainer = DistillationTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()


"""
torchrun --nproc_per_node=8 train_student_distillation.py \
    --teacher_path google-bert/bert-large-uncased-whole-word-masking-finetuned-squad \
    --student_model bert-base-uncased \
    --train_path SQuAD-v1.1.csv \
    --num_epochs 10 \
    --batch_size 16 \
    --learning_rate 3e-5 \
    --alpha 0.5 \
    --precision fp16 \
    --weight_decay 0.01 \
    --output_dir ./outputs_fp16

"""    