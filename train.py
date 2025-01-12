from transformers import AutoTokenizer, AutoModelForCausalLM,DataCollatorForSeq2Seq,Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'

base_model="C:\Jupyter\Model\Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map = 'auto',
            use_cache=False,
            attn_implementation="flash_attention_2"
)
model.gradient_checkpointing_enable()

tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # 任务类型：因果语言模型
    inference_mode=False,          # 推理模式关闭，以进行训练
    r=16,                           # 低秩值 r
    lora_alpha=32,                 # LoRA 的缩放因子
    lora_dropout=0.05,              # Dropout 概率
    use_dora=True,                # 使用DoRA
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'up_proj', 'down_proj']
)

# 将 LoRA 应用到模型中
model = get_peft_model(model, lora_config)


print(model.print_trainable_parameters())

def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {data_point["instruction"]}

                ### Input:
                {data_point["input"]}

                ### Response:
                {data_point["output"]}"""  # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}

                ### Response:
                {data_point["output"]}"""  # noqa: E501

def generate_and_tokenize_prompt(data_point):
        prompt = generate_prompt(data_point)
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=256,
            padding=False,
            return_tensors=None,
        )
        if (  # 检查最后一个 token 是否不是 EOS token，且序列长度小于 cutoff_len，并且 add_eos_token 参数为 True
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < 256
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        return result

data = load_dataset("json", data_files="C:\Jupyter\大模型\DoRA-main\commonsense_reasoning\commonsense_170k.json")

train_val = data["train"].train_test_split(test_size=1000, shuffle=True)
train_data = (train_val["train"].shuffle().map(generate_and_tokenize_prompt))
val_data = (train_val["test"].shuffle().map(generate_and_tokenize_prompt))

# 定义训练参数
training_args = TrainingArguments(
    output_dir='./results',         # 模型保存和日志输出的目录路径
    num_train_epochs=3,             # 训练的总轮数（epochs）
    per_device_train_batch_size=16, # 每个设备（如GPU或CPU）上的训练批次大小，16表示每次输入模型的数据数量
    warmup_steps=100,
    bf16=True,
    eval_strategy="steps",
    save_strategy="steps",
    optim="adamw_torch",
    save_total_limit=3,
    load_best_model_at_end=True,
    learning_rate=2e-4,             # 学习率
    logging_steps=1,               # 每隔多少步（steps）进行一次日志记录
    save_steps=500,                 # 每隔多少步保存模型
    eval_steps=500
)

# 创建 Trainer
trainer = Trainer(
    model=model,                    # 训练的模型对象，需要事先加载好
    train_dataset=train_data,
    eval_dataset=val_data,
    args=training_args,             # 上面定义的训练参数配置
    data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
)

# 开始训练
trainer.train()