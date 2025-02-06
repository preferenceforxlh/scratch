from conf import *
from util.data_loader import DataLoader


from torchtext.data.utils import get_tokenizer

# 定义分词器
tokenize_en = get_tokenizer('spacy', language='en_core_web_sm')
tokenize_de = get_tokenizer('spacy', language='de_core_news_sm')

# 初始化 DataLoader
data_loader = DataLoader(ext=('de', 'en'), tokenize_en=tokenize_en, tokenize_de=tokenize_de,
                         init_token='<sos>', eos_token='<eos>')

# 创建数据集
train_data, valid_data, test_data = data_loader.make_dataset()

# 创建迭代器
train_iter, valid_iter, test_iter = data_loader.make_iter(train_data, valid_data, test_data,
                                                                      batch_size=32, device='cuda')

enc_voc_size,dec_voc_size = data_loader.get_vocab_size()

src_pad_idx, trg_pad_idx, trg_sos_idx = data_loader.get_special_tokens_idx()