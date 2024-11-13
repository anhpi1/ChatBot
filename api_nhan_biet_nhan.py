

from huggingface_hub import InferenceClient

client = InferenceClient(api_key="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")

messages = [
	{
		"role": "user",
		"content": """
I have a series of statements and the topic labels of the following statements
1	static comparator
2	dynamic comparator
3	Hysteresis Comparator
4	2 stage comparator
5	regenerative feedback comparator
6	strong-ARM comparator
7	double tail latch type comparator
8	comparator
9	high speed comparator
10	Less than reference voltage
11	higher than reference voltage
12	OP amp
13	CMOS Comparator
14	bipolar comparator
15	Pre amp based comparator
16	Ideal comparator
17	LM339
18	LM393
19	NULL
if the statement does not belong to any of the above topics, the label of that statement is 0
please label the statements in order one by one
just number the labels without writing anything else
for example:
1
11
0
3
"""
	}
]

stream = client.chat.completions.create(
    model="Qwen/Qwen2.5-Coder-32B-Instruct", 
	messages=messages, 
	max_tokens=500,
	stream=True
)

for chunk in stream:
    print(chunk.choices[0].delta.content, end="")