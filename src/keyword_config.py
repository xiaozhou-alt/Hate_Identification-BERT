'''
此处仅仅展示了关键词的格式，完整的配置请自行通过数据集进行提取
'''
# 目标群体关键词配置
TARGET_KEYWORDS = {
    'Region': ['地域黑',],

    'Racism': ['黑人',],

    'Sexism': ['女权',],

    'LGBTQ': ['同性恋',],

    'others': ['艾滋病',]  # 不符合前四类的归为others
}

# 仇恨言论关键词
HATE_KEYWORDS = ['讨厌',]