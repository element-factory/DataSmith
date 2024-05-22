# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("rabbitcat/DataSmith-6b")

# Load model with device and dtype settings
model = AutoModelForCausalLM.from_pretrained(
    "rabbitcat/DataSmith-6b",
    device_map="auto",
    torch_dtype='auto'
).eval()

# Generate prompt and text
prompt = "读取以下文本材料，并根据材料生成问题，以及问题的答案。问题不应该是开放式的，应该能够通过材料回答。问题的答案应该在材料中表述或暗示。问题应该与材料相关，不应该太具体或太普遍。输出应为json格式。\n"
text = "文本材料：\n北京市地处中国北部、华北平原北部，东与天津市毗连，其余均与河北省相邻，中心位于东经116°20′、北纬39°56′，北京市地势西北高、东南低。西部、北部和东北部三面环山，东南部是一片缓缓向渤海倾斜的平原。境内流经的主要河流有：永定河、潮白河、北运河、拒马河等，北京市的气候为暖温带半湿润半干旱季风气候，夏季高温多雨，冬季寒冷干燥，春、秋短促。"

# Generate messages for the model
messages = [
    {
        "role": "user",
        "content": prompt + text
    }
]

# Tokenize and generate response
input_ids = tokenizer.apply_chat_template(
    conversation=messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors='pt'
)
output_ids = model.generate(
    input_ids.to('cuda'),
    max_new_tokens=512,
)
response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

print(response)
# Model response:
# [
#     {"question":"北京市位于中国的哪个方位？", "answer":"北京市地处中国北部、华北平原北部。"},
#     {"question":"北京市东临哪个城市？", "answer":"北京市东与天津市毗连。"},
#     {"question":"北京市周边与哪个省份相邻？", "answer":"北京市其余均与河北省相邻。"},
#     {"question":"北京市的中心位置是怎样的？", "answer":"北京市中心位于东经116°20′、北纬39°56′。"},
#     {"question":"北京市地势的总体特征是什么？", "answer":"北京市地势西北高、东南低。"},
#     {"question":"北京市西部、北部和东北部三面被什么环绕？", "answer":"北京市西部、北部和东北部三面环山。"},
#     {"question":"北京市东南部是什么地形？", "answer":"北京市东南部是一片缓缓向渤海倾斜的平原。"},
#     {"question":"北京市境内主要流经哪些河流？", "answer":"北京市境内流经的主要河流有永定河、潮白河、北运河、拒马河等。"},
#     {"question":"北京市的气候类型是什么？", "answer":"北京市的气候为暖温带半湿润半干旱季风气候。"},
#     {"question":"北京市哪个季节的气温最高？", "answer":"北京市夏季的气温最高。"},
#     {"question":"北京市哪个季节降雨量最多？", "answer":"北京市夏季降雨量最多。"},
#     {"question":"北京市哪个季节气候最寒冷？", "answer":"北京市冬季气候最寒冷。"},
#     {"question":"北京市哪个季节秋高气爽？", "answer":"北京市秋季气候秋高气爽。"},
#     {"question":"北京市哪个季节春暖花开？", "answer":"北京市春季气候春暖花开。"},
#     {"question":"北京市春季和秋季分别持续多长时间？", "answer":"北京市春、秋短促。"}
# ]

