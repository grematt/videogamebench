from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig, Trainer, TrainingArguments, DataCollatorWithPadding, DataCollatorForSeq2Seq
import torch
from datasets import load_dataset
import os
from src.llm.llm_client import LLMClient

import base64
import os
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Literal
import litellm
import re
from PIL import Image
from io import BytesIO

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("llm_client")

class PhiClient(LLMClient):

    def __init__(
        self, 
        model, 
        api_key: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        max_cost: float = 30.0,  # Maximum cost in USD
        log_dir: Optional[Path] = None,
        api_base: Optional[str] = None, # For Ollama 
        fine_tuned_path = ""
    ):
        super().__init__(model, api_key='no_key', temperature=temperature, max_tokens=max_tokens, max_cost=float('inf'), log_dir=log_dir, api_base=None)

        self.processor = AutoProcessor.from_pretrained(
            model, trust_remote_code = True)
 
        self.model_ref = AutoModelForCausalLM.from_pretrained(
            model if fine_tuned_path == "" else fine_tuned_path,
            # Using normal attention instead of flash attention
            _attn_implementation = "eager",
            # The new version has dtype instead of torch_dtype
            torch_dtype = torch.float16,
            trust_remote_code = True,
            )

        self.processor.tokenizer.eos_token = "<|end|>"
        self.processor.tokenizer.pad_token = "<|end|>"
        self.model_ref.config.eos_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|end|>")
        self.model_ref.config.pad_token_id = self.model_ref.config.eos_token_id
        
        self._prepare_for_inference()
        # self.model_ref.gradient_checkpointing_enable()

    def freeze_LLM_part(self):
        # Freezing the LLM part
        for name, param in self.model_ref.named_parameters():
            if "model.layers" in name or "lm_head" in name:
                param.requires_grad = False

        trainable = 0
        frozen = 0
        for name, param in self.model_ref.named_parameters():
            if param.requires_grad:
                trainable+= param.numel()
            else:
                frozen += param.numel()
                 
        print(f"Trainable params: {trainable:,}")
        print(f"Frozen params:   {frozen:,}")  

    def _prepare_for_inference(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_ref.to(self.device)

    # images is a list of image, prompt is list given by generate_response()
    def run_inference(self, images, messages):
        prompt = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            prompt,
            images
        ).to(self.device)

        self.processor.tokenizer.eos_token = "<|end|>"
        self.processor.tokenizer.pad_token = "<|end|>"
        self.model_ref.config.eos_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|end|>")
        self.model_ref.config.pad_token_id = self.model_ref.config.eos_token_id

        print('after')

        with torch.inference_mode():
            # Using KV Cache for generation
            output = self.model_ref.generate(
                **inputs,
                max_new_tokens = 200,
                # eos_token_id = self.model_ref.config.eos_token_id,
                # pad_token_id = self.model_ref.config.eos_token_id,
            )

        print('after 2')

        output = output[:, inputs["input_ids"].shape[1] :]
        clean_ids = output.masked_fill(output == -1, self.processor.tokenizer.eos_token_id)
        
        response = self.processor.tokenizer.decode(clean_ids[0], skip_special_tokens = True)

        return response

    def preprocess(self, example):

        image = example["decoded_image"]
        messages = [
            {"role": "user", "content": f"{example['question']} {example['choices']}" + "<|image_1|>\n"},
            {"role": "assistant", "content": example["answer"]}
        ]

        full_prompt = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # print(full_prompt)

        model_inputs = self.processor(
            text=full_prompt,
            images=image,
            # truncation=True,
            # padding="max_length",
            # max_length = 512
        )

        # user_input_ids = self.processor.tokenizer.apply_chat_template(
        #     messages[:-1],
        #     tokenize = True,
        #     add_generation_prompt = True
        # )

        input_ids = model_inputs["input_ids"]
        input_ids[input_ids == -1] = self.processor.tokenizer.pad_token_id
        model_inputs["input_ids"] = input_ids
        labels = model_inputs["input_ids"].clone()
 
        model_inputs["labels"] = labels
        # for key in model_inputs.keys():
        #     print(key, ": ", model_inputs[key].shape)

        model_inputs = {k: v[0] for k, v in model_inputs.items()}
        # print("Input ids: ", model_inputs["input_ids"])
        # print("Labels: ", model_inputs["labels"])

        return model_inputs

    def fine_tune(self, dataset_name, save_dir):
        # Loading the dataset
        self.load_dataset(dataset_name)
        self.freeze_LLM_part()

        processed_dataset = self.dataset.map(lambda ex: self.preprocess(ex))
        print("Dataset prorcessed...")

        # This aligns both the input_ids and the labels
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.processor.tokenizer,
            model=self.model_ref,
            padding="longest"
        ) 

        # This is to work with deepspeed and ZeRO stage 3
        training_args = TrainingArguments(
            output_dir = f"./Models/{save_dir}",
            learning_rate = 5e-5,
            num_train_epochs = 2,
            save_steps = 200,
            logging_steps = 50,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            fp16=True,
            deepspeed="ds_config.json",
        )

        trainer = Trainer(
            model=self.model_ref,
            args = training_args,
            train_dataset = processed_dataset,
            data_collator=data_collator
        )

        dl = trainer.get_train_dataloader()
        # item = next(iter(dl))
        # for key in item.keys():
        #     print(key, ": ", item[key].shape)

        # print(item["input_ids"])
        # print(item["labels"])

        # print("Start of training...")
        trainer.train()

    def load_dataset(self, dataset_name = "AI4Math/MathVista"):
        self.dataset_name = dataset_name
        dataset = load_dataset(dataset_name, split="testmini")
        self.dataset = dataset

    def benchmark(self, get_inputs, compare_outputs, dataset_name="AI4Math/MathVista"):
        
        self.load_dataset(dataset_name)
        self._prepare_for_inference()

        correct = 0
        total = 0
        for row in self.dataset:
            total+=1
            # Getting the image and the prompt from the dataset
            images, prompt = get_inputs(row)
            pred = self.run_inference(images=images, prompt=prompt).strip()
            correct+= compare_outputs(row, pred) 
            print("Accuracy: ", correct / total)

        return correct / total
    
    # ************ Overridden methods *************
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return 0.0
    
    async def generate_response(
        self, 
        system_message: Dict[str, str],
        messages: List[Dict[str, str]], 
        image_data: Optional[bytes | List[bytes]] = None
    ) -> str:
        """
        Generate a response from the language model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            image_data: Optional screenshot data to include in the prompt
            
        Returns:
            The generated response text
        """
        self.step_count += 1
        self.file_logger.info(f"Step {self.step_count} - Generating response")
        
        # Log request details
        self.file_logger.info(f"Request to model: {self.model}")
        self.file_logger.info(f"Temperature: {self.temperature}, Max tokens: {self.max_tokens}")
        
        if system_message is not None:
            messages = [system_message] + messages

        # Log the messages (excluding image data for brevity)
        messages_log = []
        img_idx = 1
        for i, msg in enumerate(messages):
            if isinstance(msg["content"], list):
                # Handle multimodal content
                text_parts = [item["text"] for item in msg["content"] if item.get("type") == "text"]
                content = ' '.join(text_parts) + f'<|image_{str(img_idx)}|>\n'
                img_idx += 1
                messages[i]['content'] = content
            else:
                content = msg["content"]
                
            messages_log.append({
                "role": msg["role"],
                "content": content[:200] + "..." if len(content) > 200 else content
            })
            
        self.file_logger.info(f"Messages: {json.dumps(messages_log, indent=2)}")

        if not isinstance(image_data, list):
            image_data = [image_data]
            
        image_data = [Image.open(BytesIO(img)) for img in image_data]

        # Generate response using litellm
        try:
            start_time = time.time()

            #response = self.run_inference(image_data, messages)
            response_text = self.run_inference(image_data, messages)
            response_time = time.time() - start_time
            
            # Log cost information
            self.file_logger.info(f"Request cost: ${0.0:.4f}")
            self.file_logger.info(f"Total cost so far: ${0.0:.2f}")
            logger.info(f"Total cost so far: ${0.0:.2f}")
            
            # Extract response text
            # response_text = response.choices[0].message.content
            
            # Log response details
            self.file_logger.info(f"Response time: {response_time:.2f}s")
            self.file_logger.info(f"Response length: {len(response_text)} characters")
            self.file_logger.info(f"Response: {response_text[:500]}...")
            
            # Write full response to a separate file
            response_file = self.log_dir / f"llm_responses.txt"
            with open(response_file, "a") as f:
                f.write(response_text)
                
            return response_text
        except Exception as e:
            self.file_logger.error(f"Error generating response: {str(e)}")
            logger.error(f"Error generating response: {str(e)}")
            return f"Error: {str(e)}"

    async def generate_react_response(
        self, 
        task: str, 
        system_message: Dict[str, str],
        history: List[Dict[str, str]], 
        screenshots: Optional[bytes | List[bytes]] = None
    ) -> Dict[str, Any]:
        """
        Generate a ReACT (Reasoning, Action, Observation) response.
        
        Args:
            task: The task description
            history: The conversation history
            screenshot: Optional screenshot data
            
        Returns:
            A dictionary containing thought, action, and action_input
        """
        self.file_logger.info(f"Step {self.step_count} - Generating ReACT response")
        self.file_logger.info(f"Task: {task}")
        
        time.sleep(5)
        
        # Create the user message with the task
        user_message = {
            "role": "user",
            "content": f"{task}"
        }
        
        # Combine messages
        messages = [system_message] + history + [user_message]
        
        # Generate response
        response_text = await self.generate_response(None, messages, screenshots)
        
        # Parse the JSON response
        try:
            import json
            # Extract JSON from the response (in case there's additional text)
            json_match = re.search(r'.*```json\s*(.*?)\s*```(?!.*```)', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response_text
                
            # Clean up any non-JSON text
            json_str = re.sub(r'^[^{]*', '', json_str)
            json_str = re.sub(r'[^}]*$', '', json_str)
            
            # Log the parsed response
            self.file_logger.info(f"Parsed json_str: {json_str}")

            response_dict = json.loads(json_str)
            
            # Ensure the response has the required fields
            required_fields = ["thought", "action", "action_input"]
            for field in required_fields:
                if field not in response_dict:
                    raise ValueError(f"Response missing required field: {field}")
            
            # Log the parsed response
            self.file_logger.info(f"Parsed ReACT response: {json.dumps(response_dict, indent=2)}")
            
            return response_dict
        except Exception as e:
            # If parsing fails, return a default response
            error_msg = f"Error parsing response: {str(e)}\nOriginal response: {response_text}"
            self.file_logger.error(error_msg)
            
            return None