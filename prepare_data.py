from tqdm import tqdm
import pandas as pd
import numpy as np
import random
from transformers import BertTokenizerFast
import json

tokenizer = BertTokenizerFast.from_pretrained('hfl/rbt3')

## 数据清洗
df = pd.read_csv('./data/sample_data.txt', sep='\t')
gps = {}
for uuid, df_gp in tqdm(df.groupby('uui_call_id'), total=len(df)):
    if (~df_gp.short).sum() < 2:
        continue
    df_gp2 = df_gp.copy(True)
    df_gp.index = range(len(df_gp))
    short_idx = df_gp[df_gp.short == 1].index
    for idx in short_idx[::-1]:
        if idx > 0 and df_gp.channel.loc[idx] == df_gp.channel.loc[idx - 1]:
            df_gp.at[idx - 1, 'content'] = df_gp.at[idx - 1, 'content'] + df_gp.at[idx, 'content']
    df_gp = df_gp.drop(short_idx)
    gps[uuid] = df_gp

# 生成数据
all_id = list(gps.keys())
df2 = pd.concat(gps.values())

mlm_data = []
seq2seq_data = []
lm_data = []
for uuid, df_gp in tqdm(df2.groupby('uui_call_id')):
    df_gp.index = range(len(df_gp))
    values = df_gp[['content', 'channel']].values
    lm = df_gp.content.iloc[0]
    for i in range(len(df_gp) - 1):
        cur = values[i: i + 2]
        text_a, channel_a = cur[0]
        text_b, channel_b = cur[1]

        # mlm数据
        if random.random() > 0.5:
            mlm_data.append(((text_a, text_b), 1))
        else:
            while 1:
                sample_id = random.choice(all_id)
                if sample_id != uuid:
                    break
            sample_df = gps[sample_id]
            sampel_text = random.choice(list(sample_df.content))
            mlm_data.append(((text_a, sampel_text), 0))

    #         # seq2seq数据
    #         if channel_a != channel_b:
    #             seq2seq_data
    to_drop = []
    for idx in df_gp.index[::-1]:
        if idx > 0 and df_gp.channel.loc[idx] == df_gp.channel.loc[idx - 1]:
            df_gp.at[idx - 1, 'content'] = df_gp.at[idx - 1, 'content'] + df_gp.at[idx, 'content']
            to_drop.append(idx)
    df_gp.drop(to_drop, inplace=True)
    content = list(df_gp.content)
    lm_data.extend(content)
    if df_gp.channel.nunique() > 1:
        seq2seq_data.extend((a, b) for a, b in zip(content[:-1], content[1:]))


# tokenize and save

def token_process(token_id):
    """以80%的几率替换为[MASK]，以10%的几率保持不变，
    以10%的几率替换为一个随机token。
    """
    rand = np.random.random()
    if rand <= 0.8:
        return tokenizer.mask_token_id
    elif rand <= 0.9:
        return token_id
    else:
        return np.random.randint(0, tokenizer.vocab_size)


def process(data, task):
    max_length = 256
    mask_rate = 0.15
    truncation = 'longest_first'
    if task == 'mlm':
        (text_a, text_b), label = data
        features = tokenizer.encode_plus(text_a, text_b, max_length=max_length, truncation=truncation)
        features['seq_label'] = label

    elif task == 'seq2seq':
        (text_a, text_b) = data
        features = tokenizer.encode_plus(text_a, text_b, max_length=max_length, truncation=truncation)
    elif task == 'lm':
        text_a = data
        features = tokenizer.encode_plus(text_a, max_length=max_length, truncation=truncation)

    input_ids = features['input_ids'][:]
    n = len(input_ids)
    mlm_label = [tokenizer.pad_token_id] * n
    rands = np.random.random(n)
    num_to_mask = max(n * mask_rate, 1)
    for idx, (rand, word) in enumerate(zip(rands, input_ids)):
        if rand < mask_rate and word not in [tokenizer.cls_token_id, tokenizer.sep_token_id] and tokenizer.decode(
                word) not in '~，。！：；【】…,.!;:':
            rand = np.random.random()
            if rand <= 0.8:
                cur_id = tokenizer.mask_token_id
            elif rand <= 0.9:
                cur_id = word
            else:
                cur_id = np.random.randint(0, tokenizer.vocab_size)
            mlm_label[idx] = cur_id
            input_ids[idx] = cur_id
    #     print(features['input_ids'], )
    #     print(input_ids)
    features['masked_input_ids'] = input_ids
    features['mlm_labels'] = mlm_label
    if task == 'mlm':
        features.pop('input_ids')
    return json.dumps(dict(features), ensure_ascii=False)


fout = open('./data/lm_data.json', 'w+')
for data in tqdm(lm_data):
    line = process(data, 'lm')
    fout.write(line+'\n')
fout.close()

fout = open('./data/mlm_data.json', 'a+')
for data in tqdm(mlm_data):
    line = process(data, 'mlm')
    fout.write(line+'\n')
fout.close()

fout = open('./data/seq2seq_data.json', 'w+')
for data in tqdm(seq2seq_data):
    line = process(data, 'seq2seq')
    fout.write(line+'\n')
fout.close()