import re
from difflib import SequenceMatcher
from keyword_config import TARGET_KEYWORDS, HATE_KEYWORDS
import json

def extract_targets(text):
    """提取评论对象(限制1-4个字)"""
    targets = []
    for group, keywords in TARGET_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                # 直接使用配置中的关键词
                if 1 <= len(kw) <= 4:
                    targets.append((kw, group))
                else:
                    # 提取1-4个字的短语
                    matches = re.findall(rf'[\u4e00-\u9fa5]{{1,4}}{kw}[\u4e00-\u9fa5]{{0,3}}|{kw}[\u4e00-\u9fa5]{{0,3}}', text)
                    targets.extend([(m, group) for m in matches if 1 <= len(m) <= 4])
    return targets

def extract_arguments(text, targets):
    """精简论点提取(优先匹配HATE_KEYWORDS)"""
    arguments = []
    
    # 1. 优先提取仇恨关键词
    for hate_kw in HATE_KEYWORDS:
        if hate_kw in text:
            arguments.append(hate_kw)
    
    # 2. 如果没有匹配到仇恨关键词，则提取目标词周围的动宾短语
    if not arguments:
        for target, _ in targets:
            pattern = rf'({target}[的]?[\u4e00-\u9fa5]{{1,3}}|[\u4e00-\u9fa5]{{1,3}}[的]?{target})'
            matches = re.findall(pattern, text)
            arguments.extend(matches)
    
    return list(set(arguments))  # 去重

def detect_hate(text, arguments):
    """检测是否仇恨言论"""
    for arg in arguments:
        if any(hate_kw in arg for hate_kw in HATE_KEYWORDS):
            return True
    return False

def generate_quads(text):
    """生成四元组"""
    targets = extract_targets(text)
    arguments = extract_arguments(text, targets)
    
    quads = []
    for target, group in targets:
        is_hate = detect_hate(text, arguments)
        arg = arguments[0] if arguments else 'NULL'
        quads.append(f"{target} | {arg} | {group} | {'hate' if is_hate else 'non-hate'}")
    
    return " [SEP] ".join(quads) + " [END]" if quads else "NULL | NULL | non-hate | non-hate [END]"

def calculate_similarity(a, b):
    """计算两个字符串的相似度"""
    return SequenceMatcher(None, a, b).ratio()

def evaluate_metrics(pred_quads, gold_quads):
    """计算硬匹配和软匹配的F1分数"""
    # 解析预测和标准答案的四元组
    pred_list = [quad.split('|') for quad in pred_quads.split('[SEP]')]
    gold_list = [quad.split('|') for quad in gold_quads.split('[SEP]')]
    
    # 初始化统计量
    tp_hard = 0  # 硬匹配真阳性
    tp_soft = 0   # 软匹配真阳性
    fp = len(pred_list)
    fn = len(gold_list)
    
    # 硬匹配计算
    for pred in pred_list:
        for gold in gold_list:
            if all(p.strip() == g.strip() for p, g in zip(pred, gold)):
                tp_hard += 1
                break
    
    # 软匹配计算
    for pred in pred_list:
        for gold in gold_list:
            # 检查Targeted Group和Hateful是否完全匹配
            if pred[2].strip() == gold[2].strip() and pred[3].strip() == gold[3].strip():
                # 计算Target和Argument的相似度
                target_sim = calculate_similarity(pred[0].strip(), gold[0].strip())
                arg_sim = calculate_similarity(pred[1].strip(), gold[1].strip())
                if target_sim > 0.5 and arg_sim > 0.5:
                    tp_soft += 1
                    break
    
    # 计算精确率、召回率和F1分数
    precision_hard = tp_hard / (tp_hard + fp) if (tp_hard + fp) > 0 else 0
    recall_hard = tp_hard / (tp_hard + fn) if (tp_hard + fn) > 0 else 0
    f1_hard = 2 * (precision_hard * recall_hard) / (precision_hard + recall_hard) if (precision_hard + recall_hard) > 0 else 0
    
    precision_soft = tp_soft / (tp_soft + fp) if (tp_soft + fp) > 0 else 0
    recall_soft = tp_soft / (tp_soft + fn) if (tp_soft + fn) > 0 else 0
    f1_soft = 2 * (precision_soft * recall_soft) / (precision_soft + recall_soft) if (precision_soft + recall_soft) > 0 else 0
    
    return {
        'hard_match': {'precision': precision_hard, 'recall': recall_hard, 'f1': f1_hard},
        'soft_match': {'precision': precision_soft, 'recall': recall_soft, 'f1': f1_soft},
        'average_f1': (f1_hard + f1_soft) / 2
    }

if __name__ == "__main__":
    # 读取test.json文件
    with open('data/test.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # 初始化统计量
    total_hard_f1 = 0
    total_soft_f1 = 0
    count = 0
    
    # 遍历测试数据
    for item in test_data:
        pred = generate_quads(item["content"])
        metrics = evaluate_metrics(pred, item["output"])
        
        print(f"ID: {item['id']}")
        print(f"输入: {item['content']}")
        print(f"预测: {pred}")
        print(f"标准: {item['output']}")
        print(f"硬匹配F1: {metrics['hard_match']['f1']:.4f}")
        print(f"软匹配F1: {metrics['soft_match']['f1']:.4f}")
        print(f"平均F1: {metrics['average_f1']:.4f}")
        print("-"*50)
        
        # 累加分数
        total_hard_f1 += metrics['hard_match']['f1']
        total_soft_f1 += metrics['soft_match']['f1']
        count += 1
    
    # 计算平均分数
    avg_hard_f1 = total_hard_f1 / count
    avg_soft_f1 = total_soft_f1 / count
    overall_avg = (avg_hard_f1 + avg_soft_f1) / 2
    
    print("\n最终评估结果:")
    print(f"平均硬匹配F1分数: {avg_hard_f1:.4f}")
    print(f"平均软匹配F1分数: {avg_soft_f1:.4f}")
    print(f"综合平均F1分数: {overall_avg:.4f}")