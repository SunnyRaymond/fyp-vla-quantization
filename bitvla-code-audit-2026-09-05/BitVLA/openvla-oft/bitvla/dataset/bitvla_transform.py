import copy
from dataclasses import dataclass
from typing import Any, Dict
from PIL import Image
import torch
from numpy import array as to_numpy_array
from transformers import LlavaProcessor
from bitvla import Bitnet_ActionTokenizer
from transformers.image_utils import get_image_size,to_numpy_array
from bitvla.constants import (
    BITNET_DEFAULT_IM_END_TOKEN,
    BITNET_DEFAULT_IMAGE_TOKEN,
    BITNET_IGNORE_INDEX,
    LLAVA_DEFAULT_IMAGE_TOKEN,
)
import re

def replace_image_tokens(input_string):
    pattern = r'\n?' + re.escape(LLAVA_DEFAULT_IMAGE_TOKEN) + r'\n?'
    replacement = BITNET_DEFAULT_IMAGE_TOKEN
    return re.sub(pattern, replacement, input_string)

def llava_to_openai(conversations):
    role_mapping = {"human": "user", "gpt": "assistant"}

    transformed_data = []
    for conversation in conversations:
        transformed_content = replace_image_tokens(conversation["value"])
        transformed_entry = {
            "role": role_mapping.get(conversation["from"], conversation["from"]),
            "content": transformed_content,
        }
        transformed_data.append(transformed_entry)

    return transformed_data

@dataclass
class BitVLA_RLDSBatchTransform:
    action_tokenizer: Bitnet_ActionTokenizer
    processor: LlavaProcessor
    predict_stop_token: bool = True
    use_wrist_image: bool = False
    use_proprio: bool = False
    end_token:str = BITNET_DEFAULT_IM_END_TOKEN
    ignore_token_idx:int = BITNET_IGNORE_INDEX
    image_token:str = BITNET_DEFAULT_IMAGE_TOKEN

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, current_action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        all_img = []
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        all_img.append(img)
        # Add additional img
        if self.use_wrist_image:
            all_wrist = []
            for k in rlds_batch["observation"].keys():
                if "wrist" in k:
                    img_wrist = Image.fromarray(rlds_batch["observation"][k][0])
                    all_wrist.append(img_wrist)
                    all_img.append(img_wrist)
        
        
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        actions = rlds_batch["action"]
        
        # Get future action chunk
        future_actions = rlds_batch["action"][1:]
        future_actions_string = ''.join(self.action_tokenizer(future_actions))

        # Get action chunk string
        current_action_string = self.action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = rlds_batch["action"].size # action_chunk_len = action_dim*action_chunk_size
        
        # source-like prompt
        sources = {
            "image": all_img,
            "conversations":[
                {
                    "from": "human",
                    "value": "<image>\n"*len(all_img)+"<proprio_pad>"+f"What action should the robot take to {lang}?"
                },
                {
                    "from": "gpt",
                    "value": action_chunk_string
                }
            ]
        }
        has_image = "image" in sources.keys()
        pixel_values = [self.processor.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0] for image in all_img]
        patch_size = self.processor.patch_size
        num_additional_image_tokens = self.processor.num_additional_image_tokens
        num_image_tokens = []
        for img in pixel_values:
            height, width = get_image_size(to_numpy_array(img))
            num_image_tokens.append((height // patch_size) * (width // patch_size))

        sources = copy.deepcopy(llava_to_openai(sources['conversations']))
        prompt = sources[0]["content"]
        x = [{"role":"user","content":prompt}]
        input_str = self.processor.tokenizer.apply_chat_template(x,tokenize=False, add_generation_prompt=True)
        input_str += sources[1]["content"]
        input_str += self.end_token
        # Replace DEFAULT_IMAGE_TOKEN with the appropriate number of tokens
        token_index = 0
        placeholder_token = "<TEMP_IMAGE_TOKEN>"
        input_str = input_str.replace(self.image_token, placeholder_token)
        while placeholder_token in input_str and token_index < len(num_image_tokens):
            input_str = input_str.replace(
                placeholder_token, 
                self.image_token * num_image_tokens[token_index], 
                1
            )
            token_index += 1
        prompt = input_str
        input_ids = self.action_tokenizer.tokenizer(prompt, add_special_tokens=True).input_ids
        labels = list(input_ids)
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        labels[: -(action_chunk_len + 1)] = self.ignore_token_idx
        if not self.predict_stop_token:
            labels[-1] = self.ignore_token_idx

        return_dict = dict(
            pixel_type = "llava", 
            pixel_value = pixel_values,
            input_ids=input_ids, 
            labels=labels, 
            dataset_name=dataset_name, 
            actions=actions)
        
        if self.use_proprio and "proprio" in rlds_batch["observation"]:
            proprio = rlds_batch["observation"]["proprio"]
            return_dict["proprio"] = proprio
        

        return return_dict