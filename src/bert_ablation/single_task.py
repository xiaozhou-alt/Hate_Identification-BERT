'''
single_task 主要修改：class HateSpeechDataset(Dataset) 类
仅预测仇恨言论的目标群体
'''

import torch
from transformers import AutoTokenizer, AutoModel
from transformers import BertModel, BertTokenizer
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
import re
from difflib import SequenceMatcher
import jieba
from hanlp_restful import HanLPClient
from keyword_config import TARGET_KEYWORDS, HATE_KEYWORDS
HanLP = HanLPClient('https://www.hanlp.com/api', auth='your_auth')  # auth需要申请

# 自定义数据集类 - 修改为支持四元组生成
class HateSpeechDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_groups = [
            'Racism', 'non-hate', 'others', 'Region', 'LGBTQ', 'Sexism',
            'Sexism,Racism', 'Sexism, others', 'Sexism, Racism,others', 'Racism,others',
            'Region,Sexism', 'Region, Racism', 'Region, others', 'LGBTQ, Region',
            'others, Sexism', 'Region, LGBTQ', 'Sexism, Region', 'Racism,Sexism',
            'LGBTQ, others', 'Sexism,Racism,Region', 'LGBTQ,Region, others', 'LGBTQ, Sexism',
            'LGBTQ, Sexism, others', 'Region,Sexism,Racism', 'Racism,Region,Sexism',
            'Sexism, LGBTQ', 'LGBTQ,Sexism,Racism', 'LGBTQ,Racism', 'Region,Racism,others',
            'LGBTQ,Racism,others'
        ]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['content']
        quads = self._parse_quads(item['output'], text)
        
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        groups = []
        for quad in quads:
            group = quad['group'] if quad['group'] in self.target_groups else 'others'
            groups.append(self.target_groups.index(group))
        
        if not quads:
            groups = [self.target_groups.index('non-hate')]
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'groups': torch.tensor(groups)
        }
    
    def _parse_quads(self, output, text):  # 添加text参数
        quad_list = []
        for quad in output.split(' [SEP] '):
            parts = quad.split(' | ')
            if len(parts) >= 4:
                target = parts[0]
                argument = parts[1]
                group = parts[2].split(',')[0].strip()
                
                # 第一层判断：保留原有的模糊匹配逻辑
                is_hate = False
                for kw in HATE_KEYWORDS:
                    ratio = SequenceMatcher(None, kw, argument).ratio()
                    if ratio >= 0.3:
                        is_hate = True
                        break
                
                # 第二层判断：如果第一层未检测到，则使用多维度特征判断
                if not is_hate:
                    features = TextProcessor.extract_hate_features(text, argument)  # 使用传入的text
                    is_hate = (
                        features['has_insult'] or
                        (features['has_negative_emoji'] and features['is_generalization']) or
                        features['has_dehumanizing']
                    )
                
                quad_list.append({
                    'target': target,
                    'argument': argument,
                    'group': group,
                    'is_hate': 1 if is_hate else 0
                })
        return quad_list
    
    @staticmethod
    def collate_fn(batch):
        # 处理变长序列
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        
        # 处理变长的groups和is_hates
        groups = [item['groups'] for item in batch]
        is_hates = [item['is_hates'] for item in batch]
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'groups': groups,  # 保持为列表形式
            'is_hates': is_hates  # 保持为列表形式
        }

# 新增文本处理工具类
class TextProcessor:
    @staticmethod
    def extract_targets(text):
        # 使用HanLP进行依存分析
        dep_result = HanLP(text, tasks='dep').get('dep')
        # 使用HanLP进行语义角色标注
        srl_result = HanLP(text, tasks='srl').get('srl')
        
        targets = []
        
        # 从SRL结果中提取ARG0（施事者）
        if isinstance(srl_result, list):
            for predicate in srl_result:
                if isinstance(predicate, dict) and 'arguments' in predicate:
                    for role in predicate['arguments']:
                        if isinstance(role, dict) and role.get('type') == 'ARG0':
                            target = role.get('word')
                            # 检查是否在TARGET_KEYWORDS中
                            for group in TARGET_KEYWORDS.values():
                                if target in group:
                                    targets.append(target)
        
        # 如果没有从SRL中找到，尝试从依存分析中找主语
        if not targets and isinstance(dep_result, list):
            for word in dep_result:
                if isinstance(word, dict) and word.get('deprel') == 'nsubj':  # 主语
                    target = word.get('lemma')
                    for group in TARGET_KEYWORDS.values():
                        if target in group:
                            targets.append(target)
        
        # 如果还是没有找到，设置为句子的主语（针对短句子）
        if not targets and isinstance(dep_result, list):
            # 查找第一个主语
            for word in dep_result:
                if isinstance(word, dict) and word.get('deprel') == 'nsubj':
                    targets.append(word.get('lemma'))
                    break
            
            # 如果没有主语，设置为null
            if not targets:
                targets.append('NULL')
        
        return list(set(targets))

    @staticmethod 
    def extract_arguments(text, targets):
        """基于SRL和依存分析提取论点（行为+受事）"""
        arguments = []
        srl_result = HanLP(text, tasks='srl').get('srl')
        
        for target in targets:
            # 查找包含该target的谓词-论元结构
            if isinstance(srl_result, list):
                for predicate in srl_result:
                    if isinstance(predicate, dict) and 'arguments' in predicate:
                        args = {}
                        for arg in predicate['arguments']:
                            if isinstance(arg, dict) and 'type' in arg and 'word' in arg:
                                args[arg['type']] = arg['word']
                        if 'ARG0' in args and args['ARG0'] == target:
                            # 组合行为和受事
                            argument = predicate.get('word', '')  # 行为
                            if 'ARG1' in args:  # 受事
                                argument += ' ' + args['ARG1']
                            arguments.append(argument)
        
        # 如果没有从SRL中找到，尝试从依存分析中提取
        if not arguments:
            dep_result = HanLP(text, tasks='dep').get('dep')
            if isinstance(dep_result, list):
                for target in targets:
                    for word in dep_result:
                        if isinstance(word, dict) and word.get('lemma') == target and word.get('deprel') == 'nsubj':
                            # 找该主语的谓语和宾语
                            head = word.get('head', 0)
                            predicate = dep_result[head-1].get('lemma', '') if head > 0 else ''
                            obj = ''
                            for w in dep_result:
                                if isinstance(w, dict) and w.get('head') == head and w.get('deprel') == 'dobj':
                                    obj = w.get('lemma', '')
                            argument = f"{predicate} {obj}".strip()
                            if argument:
                                arguments.append(argument)
        
        return arguments if arguments else [text]

    @staticmethod
    def extract_hate_features(text, argument):
        features = {
            'has_hate_keyword': any(kw in argument for kw in HATE_KEYWORDS),
            'has_insult': len(re.findall(r'[傻笨蠢贱丑懒脏坏]', argument)) > 0,
            'has_negative_emoji': len(re.findall(r'[🐶🤮🤢🤬👎😓🔪]', argument)) > 0,
            'is_generalization': any(word in argument for word in ['都', '全', '总是']),
            'has_dehumanizing': any(term in argument for term in ['东西', '货', '玩意儿'])
        }
        return features

def predict_quads(text, model, tokenizer, device='cuda'):
    processor = TextProcessor()
    targets = processor.extract_targets(text)
    arguments = processor.extract_arguments(text, targets)
    
    quads = []
    for target, arg in zip(targets, arguments):
        # 判断group
        group = 'others'
        for g, kws in TARGET_KEYWORDS.items():
            if target in kws:
                group = g
                break
        
        # 情感分析判断hate/non-hate
        sentiment = HanLP.sentiment_analysis(arg)
        is_hate = 'hate' if sentiment < 0 else 'non-hate'
        
        quads.append(f"{target} | {arg} | {group} | {is_hate}")
    
    return " [SEP] ".join(quads) + " [END]" if quads else " |  | non-hate | non-hate [END]"

# 修改训练函数以支持四元组预测
def finetune_chatglm():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 修改模型加载方式为INT4量化版本
    model_name = "/kaggle/input/bert/transformers/default/1/bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name).to(device)
    
    # 添加数据集加载
    train_dataset = HateSpeechDataset(
        '/kaggle/input/hate-identification/train.json',  # 请确认实际路径
        tokenizer,
        max_length=128
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=HateSpeechDataset.collate_fn  # 使用自定义collate函数
    )
    
    # 添加四元组预测头
    model.group_classifier = torch.nn.Linear(model.config.hidden_size, 29).cuda()  # 修改为29个类别
    
    # 修改优化器和学习率，BERT通常使用较小的学习率
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    group_criterion = nn.CrossEntropyLoss()
    hate_criterion = nn.BCEWithLogitsLoss()
    
    # 在训练循环中添加梯度累积
    gradient_accumulation_steps = 4
    
    # 训练循环
    epoch_num = 15
    
    torch.cuda.empty_cache()
    model.train()
    # 修改训练循环部分
    for epoch in range(epoch_num):
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epoch_num}')
        
        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # 前向传播
            outputs = model(input_ids, attention_mask=attention_mask)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            
            # 计算所有样本的损失
            total_loss = 0
            batch_group_loss = 0
            
            # 处理每个样本的四元组
            for i in range(len(batch['groups'])):
                # 获取当前样本的预测和标签
                group_logits = model.group_classifier(cls_embedding[i].unsqueeze(0))
                
                # 确保标签形状匹配
                group_labels = batch['groups'][i].to(device).long()
                
                # 计算损失
                group_loss = group_criterion(
                    group_logits.repeat(group_labels.size(0), 1),
                    group_labels
                )
                
                total_loss += group_loss / gradient_accumulation_steps
            
            total_loss.backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            progress_bar.set_postfix({'loss': total_loss.item()})
    
    # 保存完整模型
    print("模型已保存！")
    torch.save(model.state_dict(), '/kaggle/working/bert_model.pth')

if __name__ == "__main__":
    finetune_chatglm()