"""
LoRA微调 BERT-large for SQuAD QA

用法:
torchrun --nproc_per_node=8 train_lora_finetune.py \
    --model_path google-bert/bert-large-uncased-whole-word-masking-finetuned-squad \
    --train_path SQuAD-v1.1.csv \
    --num_epochs 5 \
    --batch_size 16 \
    --learning_rate 3e-4 \
    --lora_r 8 \
    --lora_alpha 16 \
    --precision fp32 \
    --output_dir ./outputs_lora
"""

import os
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    get_cosine_schedule_with_warmup,
    set_seed
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import argparse
import pandas as pd
import json

class SQuADQADataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    
    def __len__(self):
        return len(self.encodings['input_ids'])
    
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

class LoRATrainer:
    def __init__(self, args):
        self.args = args
        self.setup_distributed()
        self.setup_model_and_data()
        self.setup_training()
        
    def setup_distributed(self):
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
            print(f" LoRA微调 BERT-large for SQuAD")
            print(f"{'='*70}")
            print(f" 分布式设置: World Size={self.world_size}, Rank={self.rank}")
    
    def load_squad_qa_data(self, csv_path):
        if self.is_main:
            print(f"\n 加载数据: {csv_path}")
        
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=['context', 'question', 'answer'])
        
        if self.is_main:
            print(f"   - 有效样本: {len(df):,}")
        
        return df['context'].tolist(), df['question'].tolist(), df['answer'].tolist()
    
    def prepare_qa_features(self, contexts, questions, answers, tokenizer):
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
        # 加载tokenizer和数据
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path)
        
        train_contexts, train_questions, train_answers = self.load_squad_qa_data(
            self.args.train_path
        )
        train_encodings = self.prepare_qa_features(
            train_contexts, train_questions, train_answers, self.tokenizer
        )
        
        # 划分训练/验证集
        train_size = int(len(train_encodings['input_ids']) * 0.9)
        train_enc = {k: v[:train_size] for k, v in train_encodings.items()}
        eval_enc = {k: v[train_size:] for k, v in train_encodings.items()}
        
        self.train_dataset = SQuADQADataset(train_enc)
        self.eval_dataset = SQuADQADataset(eval_enc)
        
        if self.is_main:
            print(f"   - 训练集: {len(self.train_dataset):,}")
            print(f"   - 验证集: {len(self.eval_dataset):,}")
        
        # 加载模型并应用LoRA
        if self.is_main:
            print(f"\n 加载模型: {self.args.model_path}")
        
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            self.args.model_path
        ).to(self.local_rank)
        
        # 配置LoRA
        lora_config = LoraConfig(
            task_type=TaskType.QUESTION_ANS,
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            target_modules=["query", "value"],  # BERT的attention层
            bias="none"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        if self.is_main:
            self.model.print_trainable_parameters()
        
        # DDP包装
        if self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank
            )
    
    def setup_training(self):
        # DataLoader
        train_sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True
        ) if self.world_size > 1 else None
        
        eval_sampler = DistributedSampler(
            self.eval_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False
        ) if self.world_size > 1 else None
        
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
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        
        # 学习率调度器
        num_training_steps = len(self.train_loader) * self.args.num_epochs
        num_warmup_steps = int(num_training_steps * 0.1)
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Mixed Precision
        self.use_amp = False
        self.scaler = None
        self.amp_dtype = torch.float32
        
        if self.args.precision == "fp16":
            self.use_amp = True
            self.amp_dtype = torch.float16
            self.scaler = GradScaler()
        elif self.args.precision == "bf16" and torch.cuda.is_bf16_supported():
            self.use_amp = True
            self.amp_dtype = torch.bfloat16
        
        if self.is_main:
            print(f"\n  训练配置:")
            print(f"   - LoRA rank: {self.args.lora_r}")
            print(f"   - LoRA alpha: {self.args.lora_alpha}")
            print(f"   - Precision: {self.args.precision}")
            print(f"   - Batch Size: {self.args.batch_size}")
            print(f"   - Global Batch Size: {self.args.batch_size * self.world_size}")
            print(f"   - Learning Rate: {self.args.learning_rate}")
            print(f"   - Training Steps: {num_training_steps:,}")
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        if self.world_size > 1:
            self.train_loader.sampler.set_epoch(epoch)
        
        pbar = tqdm(self.train_loader, disable=not self.is_main, 
                   desc=f"Epoch {epoch+1}/{self.args.num_epochs}")
        
        for batch in pbar:
            batch = {k: v.to(self.local_rank) for k, v in batch.items()}
            
            if self.use_amp:
                with autocast(device_type='cuda', dtype=self.amp_dtype):
                    outputs = self.model(**batch)
                    loss = outputs.loss
            else:
                outputs = self.model(**batch)
                loss = outputs.loss
            
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            total_loss += loss.item()
            
            if self.is_main:
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
                })
        
        return total_loss / len(self.train_loader)
    
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_loader, disable=not self.is_main, desc="Eval"):
                batch = {k: v.to(self.local_rank) for k, v in batch.items()}
                
                if self.use_amp:
                    with autocast(device_type='cuda', dtype=self.amp_dtype):
                        outputs = self.model(**batch)
                        loss = outputs.loss
                else:
                    outputs = self.model(**batch)
                    loss = outputs.loss
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.eval_loader)
        
        if self.world_size > 1:
            avg_loss_tensor = torch.tensor(avg_loss).to(self.local_rank)
            dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss_tensor.item() / self.world_size
        
        return avg_loss
    
    def save_checkpoint(self, epoch, eval_loss, is_best=False):
        if not self.is_main:
            return
        
        model = self.model.module if self.world_size > 1 else self.model
        
        checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-epoch-{epoch+1}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        print(f"    保存checkpoint: {checkpoint_dir}")
        
        if is_best:
            best_model_dir = os.path.join(self.args.output_dir, "best_model")
            os.makedirs(best_model_dir, exist_ok=True)
            
            model.save_pretrained(best_model_dir)
            self.tokenizer.save_pretrained(best_model_dir)
            
            print(f"    保存最佳模型: {best_model_dir}")
    
    def train(self):
        start_time = time.time()
        best_eval_loss = float('inf')
        
        for epoch in range(self.args.num_epochs):
            if self.is_main:
                print(f"\n{'='*70}")
                print(f" Epoch {epoch+1}/{self.args.num_epochs}")
                print(f"{'='*70}")
            
            train_loss = self.train_epoch(epoch)
            eval_loss = self.evaluate()
            
            is_best = eval_loss < best_eval_loss
            if is_best:
                best_eval_loss = eval_loss
            
            if self.is_main:
                print(f"\n 结果:")
                print(f"   - Train Loss: {train_loss:.4f}")
                print(f"   - Eval Loss:  {eval_loss:.4f} {' (BEST)' if is_best else ''}")
            
            self.save_checkpoint(epoch, eval_loss, is_best=is_best)
        
        total_time = time.time() - start_time
        
        if self.is_main:
            print(f"\n{'='*70}")
            print(f" 训练完成!")
            print(f"{'='*70}")
            print(f"   - 总耗时: {total_time/3600:.2f} 小时")
            print(f"   - 最佳Eval Loss: {best_eval_loss:.4f}")
            print(f"   - 最佳模型: {os.path.join(self.args.output_dir, 'best_model')}")
            print(f"{'='*70}\n")
        
        if self.world_size > 1:
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description='LoRA微调 BERT-large for SQuAD')
    
    # 数据和模型
    parser.add_argument('--model_path', type=str, required=True,
                       help='预训练模型路径')
    parser.add_argument('--train_path', type=str, required=True,
                       help='训练数据CSV路径')
    parser.add_argument('--max_length', type=int, default=512,
                       help='最大序列长度')
    
    # LoRA参数
    parser.add_argument('--lora_r', type=int, default=8,
                       help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16,
                       help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                       help='LoRA dropout')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='权重衰减')
    parser.add_argument('--precision', type=str, default='fp32',
                       choices=['fp32', 'fp16', 'bf16'],
                       help='训练精度')
    
    # 输出
    parser.add_argument('--output_dir', type=str, default='./outputs_lora',
                       help='输出目录')
    
    args = parser.parse_args()
    
    set_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)
    
    trainer = LoRATrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()