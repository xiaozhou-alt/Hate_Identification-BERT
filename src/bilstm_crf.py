import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer
from difflib import SequenceMatcher

# 数据预处理类
class HateSpeechDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=128):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['content']
        
        # 处理多个四元组的情况
        quad_list = []
        output_str = item['output'].replace('[END]', '').strip()
        
        if '[SEP]' in output_str:
            # 多个四元组的情况
            quads = output_str.split('[SEP]')
            for quad in quads:
                parts = quad.strip().split(' | ')
                if len(parts) >= 4:
                    quad_list.append({
                        'target': parts[0],
                        'argument': parts[1],
                        'group': parts[2],
                        'label': parts[3]
                    })
        else:
            # 单个四元组的情况
            parts = output_str.split(' | ')
            if len(parts) >= 4:
                quad_list.append({
                    'target': parts[0],
                    'argument': parts[1],
                    'group': parts[2],
                    'label': parts[3]
                })
        
        # 默认取第一个四元组作为训练数据
        target = quad_list[0]['target'] if quad_list else ''
        argument = quad_list[0]['argument'] if quad_list else ''
        group = quad_list[0]['group'] if quad_list else ''
        label = quad_list[0]['label'] if quad_list else 'non-hate'
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'target': target,
            'argument': argument,
            'group': group,
            'label': 1 if label != 'non-hate' else 0
        }

# BiLSTM-CRF模型
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags):
        super(BiLSTM_CRF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                             num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, num_tags)
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.num_tags = num_tags
        
    def forward(self, x, mask):
        embedded = self.embedding(x)
        lstm_out, _ = self.bilstm(embedded)
        emissions = self.hidden2tag(lstm_out)
        return emissions
    
    def _score_sentence(self, emissions, tags, mask):
        score = torch.zeros(1).to(emissions.device)
        tags = torch.cat([torch.tensor([0], dtype=torch.long).to(emissions.device), tags])
        
        for i, emission in enumerate(emissions):
            if mask[i]:
                score = score + self.transitions[tags[i + 1], tags[i]] + emission[tags[i + 1]]
        
        return score
    
    def _viterbi_decode(self, emissions, mask):
        length = mask.sum()
        emissions = emissions[:length]
        
        viterbi = torch.zeros(self.num_tags).to(emissions.device)
        backpointers = []
        
        for emission in emissions:
            viterbi = viterbi.unsqueeze(0) + self.transitions + emission.unsqueeze(1)
            best_scores, best_tags = viterbi.max(1)
            backpointers.append(best_tags)
            viterbi = best_scores
        
        best_score, best_tag = viterbi.max(0)
        best_tags = [best_tag.item()]
        
        for bp in reversed(backpointers[1:]):
            best_tags.append(bp[best_tags[-1]].item())
        
        best_tags.reverse()
        return best_score, best_tags

# 训练和评估函数
def train_and_evaluate():
    # 初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 修改这行代码
    model_name = "./model/bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # 准备数据
    train_dataset = HateSpeechDataset('./data/train.json', tokenizer)
    test_dataset = HateSpeechDataset('./data/test.json', tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # 初始化模型
    model = BiLSTM_CRF(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=100,
        hidden_dim=256,
        num_tags=2  # hate/non-hate
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    epoch_num = 10

    # 训练
    for epoch in range(epoch_num):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            emissions = model(input_ids, attention_mask)
            # 获取序列中第一个token的预测（[CLS] token）作为分类结果
            cls_emissions = emissions[:, 0, :]  # 取每个序列的第一个token
            loss = criterion(cls_emissions, labels)  # 直接使用labels，无需view
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}')

    # 模型保存
    torch.save(model.state_dict(), './model/bilstm/bilstm.pth')
    
    # 评估
    model.eval()
    predictions, true_labels = [], []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc='Evaluating')):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            emissions = model(input_ids, attention_mask)
            cls_emissions = emissions[:, 0, :]
            _, preds = torch.max(cls_emissions, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    # 使用新的评估方法
    hard_f1, soft_f1, avg_f1 = evaluate_predictions(true_labels, predictions, test_dataset)
    
    return hard_f1, soft_f1, avg_f1

def calculate_similarity(pred, gold):
    """计算两个字符串的相似度"""
    return SequenceMatcher(None, pred, gold).ratio()

def evaluate_predictions(true_labels, predictions, test_data):
    """
    计算硬匹配和软匹配的F1分数
    :param true_labels: 真实标签列表
    :param predictions: 预测标签列表
    :param test_data: 测试数据集
    """
    hard_match = []
    soft_match = []
    
    for idx, (pred, gold) in enumerate(zip(predictions, true_labels)):
        # 硬匹配
        hard_correct = (pred == gold)
        hard_match.append(hard_correct)
        
        # 软匹配
        # 当前版本只做简单的标签匹配
        soft_correct = (pred == gold)
        soft_match.append(soft_correct)
    
    # 计算F1分数
    hard_f1 = f1_score(true_labels, predictions, average='weighted')
    soft_f1 = f1_score([int(x) for x in soft_match], 
                      [1]*len(soft_match), average='weighted')
    
    avg_f1 = (hard_f1 + soft_f1) / 2
    
    print(f'Hard Match F1: {hard_f1:.4f}')
    print(f'Soft Match F1: {soft_f1:.4f}')
    print(f'Average F1: {avg_f1:.4f}')
    
    return hard_f1, soft_f1, avg_f1

def predict_and_compare(model, test_dataset, num_samples=5):
    """
    随机抽取测试数据并输出预测结果对比
    :param model: 训练好的模型
    :param test_dataset: 测试数据集
    :param num_samples: 要抽取的样本数量
    """
    import random
    
    # 随机选择样本索引
    sample_indices = random.sample(range(len(test_dataset)), num_samples)
    
    print("="*50)
    print(f"随机抽取{num_samples}条测试数据对比:")
    print("="*50)
    
    for idx in sample_indices:
        item = test_dataset[idx]
        # 修改这行代码，使用正确的字段名
        text = test_dataset.data[idx]['content']  # 直接从原始数据获取
        
        # 获取真实标签
        true_label = item['label']
        
        # 准备模型输入
        device = next(model.parameters()).device  # 获取模型所在的设备
        input_ids = item['input_ids'].unsqueeze(0).to(device)
        attention_mask = item['attention_mask'].unsqueeze(0).to(device)
        
        # 预测
        with torch.no_grad():
            emissions = model(input_ids, attention_mask)
            cls_emissions = emissions[:, 0, :]
            _, pred_label = torch.max(cls_emissions, dim=1)
            pred_label = pred_label.item()
        
        # 获取四元组信息
        output_str = test_dataset.data[idx]['output']
        
        print(f"\n原句: {text}")
        print(f"真实output: {output_str}")
        print(f"预测标签: {'hate' if pred_label == 1 else 'non-hate'}")
        print(f"真实标签: {'hate' if true_label == 1 else 'non-hate'}")
        print("-"*50)

# 在train_and_evaluate函数末尾添加调用
if __name__ == '__main__':
    hard_f1, soft_f1, avg_f1 = train_and_evaluate()
    
    # 初始化模型和测试集
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained("./model/bert-base-chinese")
    test_dataset = HateSpeechDataset('./data/test.json', tokenizer)
    
    model = BiLSTM_CRF(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=100,
        hidden_dim=256,
        num_tags=2
    ).to(device)
    
    # 加载训练好的模型权重
    model.load_state_dict(torch.load('./model/bilstm/bilstm.pth'))  # 假设模型权重已保存
    
    # 调用预测对比函数
    predict_and_compare(model, test_dataset)