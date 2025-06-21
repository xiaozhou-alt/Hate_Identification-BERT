def predict_batch(model_path='/kaggle/working/bert_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型和tokenizer
    tokenizer = BertTokenizer.from_pretrained("/kaggle/input/bert/transformers/default/1/bert-base-chinese")
    model = BertModel.from_pretrained("/kaggle/input/bert/transformers/default/1/bert-base-chinese").to(device)
    
    # 添加预测头并加载权重
    model.group_classifier = torch.nn.Linear(model.config.hidden_size, 6).to(device)
    model.hate_classifier = torch.nn.Linear(model.config.hidden_size, 1).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 读取test.json文件
    with open('/kaggle/input/hate-identification/test.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    target_groups = ['Region', 'Racism', 'Sexism', 'LGBTQ', 'others', 'non-hate']
    processor = TextProcessor()
    
    for item in test_data[:20]:  # 只展示前20个样本
        text = item['content']
        true_output = item['output']
        
        # 文本预处理
        inputs = tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            
            # 获取预测结果
            group_logits = model.group_classifier(cls_embedding)
            hate_logits = model.hate_classifier(cls_embedding)
            
            group_pred = torch.argmax(group_logits, dim=1).item()
            hate_pred = torch.sigmoid(hate_logits).item() > 0.5
            
            # 获取目标组名
            group_name = target_groups[group_pred]
            
            # 提取目标和论点
            targets = processor.extract_targets(text)
            arguments = processor.extract_arguments(text, targets)
            
            # 生成四元组
            quads = []
            for target, arg in zip(targets, arguments):
                quads.append(f"{target} | {arg} | {group_name} | {'hate' if hate_pred else 'non-hate'}")
            
            pred_output = " [SEP] ".join(quads) if quads else "NULL | NULL | non-hate | non-hate"
        
        print(f"Content: {text}")
        print(f"True Output: {true_output}")
        print(f"Pred Output: {pred_output}")
        print("--"*50)

if __name__ == "__main__":
    predict_batch()