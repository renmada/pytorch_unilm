# pytorch_unilm
pretrain unilm for specific domain

## Background
由于业务需求，需要做通话录音内容做分析。主要分为两个任务：
1. 识别通话中客户的投诉
2. 对通话内容做20个字内的小结
```
通话：['前台。', '帮我们拿个凳子上来可以吗？', '换凳子。', '凳子。', '怎么没有吗？这是这个凳子吗？', '我们吃饭怎么吃啊？', '这是你的凳子拿，', '您是哪个房间？', '828216 。']
小结：8286房间需要一个凳子
```
为获得更好的效果，打算做领域的训练unilm

## 数据处理
unilm的三种类型的任务
### mlm
从一个对话中按顺序选择一条作为text_a, 50%概率取下一条，此时nsp_label为1，50%概率从语料取一句，nsp_label为0
### lm
合并连续同一个声道文本, 作为text_a
### seq2seq
合并连续同一个声道文本, 取两个不同声道作为text_a, text_b
![pic/sample.png]

## Usage
按sample的格式准备好数据
```python
python prepare_data.py
python run_pretrain.py
```






