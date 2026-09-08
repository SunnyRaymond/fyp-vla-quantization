import json
import os
import sys

CKPT_PTH = sys.argv[1]


# 1. add vocab_size to config.json
config_path = os.path.join(CKPT_PTH, "config.json")
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

if "text_config" in config and "vocab_size" in config["text_config"]:
    vocab_size = config["text_config"]["vocab_size"]
else:
    raise KeyError("vocab_size not found in config['text_config']")

config["vocab_size"] = vocab_size
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
    
    
# 2. change the tokenizer-related files
special_tokens_map_pth = os.path.join(CKPT_PTH, "special_tokens_map.json")
tokenizer_config_pth = os.path.join(CKPT_PTH, "tokenizer_config.json")
tokenizer_pth = os.path.join(CKPT_PTH, "tokenizer.json")

# a. make "additional_special_tokens" as a empty list
with open(special_tokens_map_pth, "r", encoding="utf-8") as f:
    special_tokens_map = json.load(f)
special_tokens_map["additional_special_tokens"] = []
with open(special_tokens_map_pth, "w", encoding="utf-8") as f:
    json.dump(special_tokens_map, f, indent=2, ensure_ascii=False)
    
# b. in "added_tokens_decoder", change "128002"'s "content" to "<|pad|>"
#                               delete all the keys after "128009"
#    in "additional_special_tokens",make it as a empty list
#    for "chat_template", change it to add system prompt
with open(tokenizer_config_pth, "r", encoding="utf-8") as f:
    tokenizer_config = json.load(f)
added_tokens_decoder = {}
for k, v in tokenizer_config["added_tokens_decoder"].items():
    if int(k) > 128009:
        continue
    added_tokens_decoder[k] = v
added_tokens_decoder["128002"]["content"] = "<|pad|>"
tokenizer_config["added_tokens_decoder"] = added_tokens_decoder

tokenizer_config["additional_special_tokens"] = []
tokenizer_config["chat_template"] = "System: A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.<|eot_id|>{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = message['role'] | capitalize + ': '+ message['content'] | trim + '<|eot_id|>' %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant: ' }}{% endif %}"
with open(tokenizer_config_pth, "w", encoding="utf-8") as f:
    json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
    
# c. in "added_tokens" which is a list, if "id == 128002", change "content" to "<|pad|>"
#                                       delete all the tokens after "id == 128009"
with open(tokenizer_pth, "r", encoding="utf-8") as f:
    tokenizer = json.load(f)
added_tokens = []
for token in tokenizer["added_tokens"]:
    if token["id"] > 128009:
        continue
    if token["id"] == 128002:
        token["content"] = "<|pad|>"
    added_tokens.append(token)
tokenizer["added_tokens"] = added_tokens
with open(tokenizer_pth, "w", encoding="utf-8") as f:
    json.dump(tokenizer, f, indent=2, ensure_ascii=False)
