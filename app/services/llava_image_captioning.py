import structlog
import torch
import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

import deepl
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


logger = structlog.get_logger()


class llavaImageCaptioning:
    def __init__(self, deepl_key, **kwargs):
        self.args = kwargs
        
        # Model
        disable_torch_init()

        model_name = get_model_name_from_path(kwargs.get("model_path"))
        self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model(
            kwargs.get("model_path"), 
            kwargs.get("model_base"), 
            model_name, 
            kwargs.get("load_8bit"), 
            kwargs.get("load_4bit"), 
            device=kwargs.get("device"), 
        )

        if 'llama-2' in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        if conv_mode is not None and conv_mode != conv_mode:
            logger.error('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, conv_mode, conv_mode))
        else:
            conv_mode = conv_mode

        conv = conv_templates[conv_mode].copy()
        
        if "mpt" in model_name.lower():
            roles = ('user', 'assistant')
        else:
            roles = conv.roles
            
        prompt = kwargs.get("prompt")
        inp = f"{roles[0]}: {prompt}"

        if self.model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        self.input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        self.stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, self.input_ids)
        self.streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        self.deepl_key = deepl_key

    def _caption_translate(self, en_caption):
        translator = deepl.Translator(self.deepl_key)
        kr_caption = translator.translate_text(en_caption, target_lang="ko")
        
        logger.info("Translating English caption to Korean", english_caption=en_caption, translated_caption=kr_caption.text)
        
        return kr_caption.text
    
    def _load_image(self, image_file):
        if image_file.startswith('http://') or image_file.startswith('https://'):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert('RGB')
            
            logger.info("Fetching and processing image from URL", image_url=image_file)
        else:
            image = Image.open(image_file).convert('RGB')
            
            logger.info("Loading and processing local image file", image_path=image_file)
        
        return image
    
    def run(self, image_file):
        image = self._load_image(image_file)
        
        # Similar operation in model_worker.py
        image_tensor = process_images([image], self.image_processor, self.args)
        if type(image_tensor) is list:
            image_tensor = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

        with torch.inference_mode():
            output_ids = self.model.generate(
                self.input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=self.args.get("temperature"),
                max_new_tokens=self.args.get("max_new_tokens"),
                streamer=self.streamer,
                use_cache=True,
                stopping_criteria=[self.stopping_criteria])

        outputs = self.tokenizer.decode(output_ids[0, self.input_ids.shape[1]:]).strip()
        outputs = outputs.replace('</s>', '').strip()
        
        logger.info("Generated caption from LLaVA model", outputs=outputs)
        
        return self._caption_translate(outputs)