# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import torch
import gc
import numpy as np
from omegaconf import OmegaConf
import uuid
from tqdm import tqdm
import random
from einops import rearrange
import soundfile as sf
from transformers import AutoModelForCausalLM, LogitsProcessorList,BitsAndBytesConfig
from .inference.mmtokenizer import _MMSentencePieceTokenizer
from .inference.codecmanipulator import CodecManipulator
from .inference.infer import load_audio_mono,encode_audio,split_lyrics,BlockTokenRangeProcessor,stage2_inference,save_audio,seed_everything
from .inference.xcodec_mini_infer.vocoder import build_codec_model,process_audio
from .inference.xcodec_mini_infer.post_process_audio import replace_low_freq_with_energy_matched
from .inference.xcodec_mini_infer.models.soundstream_hubert_new import SoundStream
# from .inference.infer_stage1 import Stage1Pipeline_EXL2,SampleSettings,Stage1Pipeline_HF
# from .inference.infer_stage2 import Stage2Pipeline_EXL2 ,Stage2Pipeline_HF
# from .inference.infer_postprocess import post_process
# from mmgp import offload
import folder_paths


MAX_SEED = np.iinfo(np.int32).max
current_node_path = os.path.dirname(os.path.abspath(__file__))
device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# add checkpoints dir
YUE_weigths_path = os.path.join(folder_paths.models_dir, "yue")
if not os.path.exists(YUE_weigths_path):
    os.makedirs(YUE_weigths_path)
folder_paths.add_model_folder_path("yue", YUE_weigths_path)


class YUE_Stage_A_Loader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        ckpt_list_xcodec = [i for i in folder_paths.get_filename_list("yue") if "36" in i]
        return {
            "required": {
                "stage_A_repo": ("STRING",{"default": "YuE-s1-7B-anneal-en-cot", "tooltip": "Stage A模型仓库名称，用于音乐生成的第一阶段"}),
                "xcodec_ckpt": (["none"] + ckpt_list_xcodec, {"tooltip": "XCodec检查点文件，用于音频编解码"}),
                "quantization_model":(["fp16","int8","int4","exllamav2"], {"tooltip": "模型量化方式：fp16(半精度)、int8(8位整数)、int4(4位整数)、exllamav2(高效量化)"}),
                "use_mmgp":("BOOLEAN",{"default":True, "tooltip": "是否使用MMGP(多模态生成预处理)，建议开启以获得更好的生成质量"}),
                "stage1_cache_size": ("INT",{"default": 65536, "min": 8192, "max": MAX_SEED, "step": 64, "display": "number", "tooltip": "Stage1缓存大小，影响推理速度和内存占用。更大的值可提升性能但占用更多内存"}),
                "exllamav2_cache_mode": (["FP16","Q8","Q6", "Q4"], {"tooltip": "ExLlamaV2缓存模式：FP16(最高质量)、Q8(8位量化)、Q6(6位量化)、Q4(4位量化，最省内存)"}),
                "mmgp_profile": ([0,1,2,3,4,5], {"tooltip": "MMGP配置文件编号(0-5)，不同配置对应不同的生成策略和质量"}),
            },
        }

    RETURN_TYPES = ("MODEL_YUE_A",)
    RETURN_NAMES = ("model",)
    FUNCTION = "loader_main"
    CATEGORY = "YUE"

    def loader_main(self, stage_A_repo, xcodec_ckpt,quantization_model,use_mmgp,stage1_cache_size,exllamav2_cache_mode,mmgp_profile):
        # Debug: Print original stage_A_repo value
        print(f"[DEBUG] Original stage_A_repo: {stage_A_repo}")
        print(f"[DEBUG] YUE_weigths_path: {YUE_weigths_path}")
        
        # Convert stage_A_repo to full local path
        original_stage_A_repo = stage_A_repo
        local_model_path = os.path.join(YUE_weigths_path, stage_A_repo)
        
        # Debug: Print path construction
        print(f"[DEBUG] Constructed local path: {local_model_path}")
        print(f"[DEBUG] Local path exists: {os.path.exists(local_model_path)}")
        
        # Check if local model folder exists
        if os.path.exists(local_model_path) and os.path.isdir(local_model_path):
            stage_A_repo = local_model_path
            print(f"[DEBUG] Using local model path: {stage_A_repo}")
        else:
            print(f"[ERROR] Local model folder '{local_model_path}' not found")
            print(f"[DEBUG] Available files in yue folder: {folder_paths.get_filename_list('yue')}")
            # Fallback to original value for Hugging Face download
            stage_A_repo = f"m-a-p/{original_stage_A_repo}"
            print(f"[DEBUG] Falling back to Hugging Face repo: {stage_A_repo}")
        
        basic_model_config=os.path.join(current_node_path, "inference/xcodec_mini_infer/final_ckpt/config.yaml")
        model_config = OmegaConf.load(basic_model_config)
        resume_path=folder_paths.get_full_path("yue", xcodec_ckpt)

        if quantization_model=="exllamav2":
            from .inference.infer_stage1 import Stage1Pipeline_EXL2,Stage1Pipeline_HF
            
            torch.autograd.grad_mode._enter_inference_mode(True)
            torch.autograd.set_grad_enabled(False)
            if exllamav2_cache_mode=="FP16":
                print("**********Loading HF,fp16*********")
                stage_1_model = Stage1Pipeline_HF(
                        model_path=stage_A_repo,
                        device=device,
                        basic_model_config=basic_model_config,
                        resume_path=resume_path,
                        cache_size=stage1_cache_size,
                    )
            else:
                print("**********Loading EXLLAMA V2*********")
                stage_1_model = Stage1Pipeline_EXL2(
                    model_path=stage_A_repo,
                    device=device,
                    basic_model_config=basic_model_config,
                    resume_path=resume_path,
                    cache_size=stage1_cache_size,
                    cache_mode=exllamav2_cache_mode,
                    )
            codectool_stage2=None
            codectool=None
            mmtokenizer=None
            assert model_config.generator.name == "SoundStream"
            codec_model = SoundStream(**model_config.generator.config).to(device)
            parameter_dict = torch.load(resume_path, map_location=device, weights_only=False)
            codec_model.load_state_dict(parameter_dict["codec_model"])
            codec_model.eval()
        else:
            mmtokenizer = _MMSentencePieceTokenizer(os.path.join(current_node_path, "inference/mm_tokenizer_v0.2_hf/tokenizer.model"))
            if quantization_model=="int8":
                # Load 8-bit quantized model using bitsandbytes
                print("**********Loading int8*********")
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                stage_1_model = AutoModelForCausalLM.from_pretrained(
                    stage_A_repo, 
                    torch_dtype=torch.bfloat16,
                    quantization_config=quantization_config,
                    attn_implementation="flash_attention_2",  # To enable flashattn, you have to install flash-attn
                )
            elif quantization_model=="int4":
                # Load 4-bit quantized model using bitsandbytes
                print("**********Loading int4*********")
                quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="int4"
                        )
                stage_1_model = AutoModelForCausalLM.from_pretrained(
                    stage_A_repo, 
                    torch_dtype=torch.bfloat16,
                    quantization_config=quantization_config,
                    attn_implementation="flash_attention_2",)         
            else:
                # Load model without quantization
                
                stage_1_model = AutoModelForCausalLM.from_pretrained(
                    stage_A_repo, 
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",  # To enable flashattn, you have to install flash-attn
                )
        

            if torch.__version__ >= "2.0.0":
                stage_1_model = torch.compile(stage_1_model,mode="max-autotune")
            # tokenizer = AutoTokenizer.from_pretrained(os.path.join(current_node_path, "mm_tokenizer_v0.2_hf"))
            # stage_1_model = AutoModel.from_pretrained(folder_paths.get_full_path("yue", stage_A))
            # to device, if gpu is available
            if use_mmgp:
                print("**********Loading mmpg *********")
                stage_1_model.to("cpu")
            else:
                print("**********Loading fp16*********")
                stage_1_model = stage_1_model.to('cuda:0')
    
            stage_1_model.eval()

            if use_mmgp:
                from mmgp import offload
                pipe = { "transformer": stage_1_model,}
                kwargs  = {}
                if mmgp_profile == 4 :
                    kwargs["budgets"] =  { "transformer": 3000, "*" : 5000 }
                elif mmgp_profile == 2:
                    kwargs["budgets"] =  5000
                compile= True
                quantizeTransformer = mmgp_profile == 3 or mmgp_profile == 4 or mmgp_profile == 5 
                offload.profile(pipe, profile_no = mmgp_profile,   quantizeTransformer= quantizeTransformer, compile = compile, verboseLevel= 1, **kwargs )
         
            codectool = CodecManipulator("xcodec", 0, 1)
            codectool_stage2 = CodecManipulator("xcodec", 0, 8)
            
            codec_model = eval(model_config.generator.name)(**model_config.generator.config).to(device)
            parameter_dict = torch.load(resume_path, map_location='cpu', weights_only=False)
            codec_model.load_state_dict(parameter_dict['codec_model'])
            codec_model.to(device)
            codec_model.eval()

        torch.cuda.empty_cache()
        gc.collect()
        
        return ({"stage_1_model":stage_1_model,"codectool_stage2":codectool_stage2,"codectool":codectool,"codec_model":codec_model,"mmtokenizer":mmtokenizer,"quantization_model":quantization_model,"mmgp_profile":mmgp_profile},)


class YUE_Stage_A_Sampler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL_YUE_A",),
                "genres_prompt": ("STRING", {"default": "inspiring female uplifting pop airy vocal electronic bright vocal vocal.", "multiline": False, "tooltip": "音乐风格描述，支持乐器、流派、情绪、人声音色和性别等标签"}),
                "lyrics_prompt": ("STRING", {"default":
                    "[verse]\nStaring at the sunset, colors paint the sky.\nThoughts of you keep swirling, can't deny.\nI know I let you down, I made mistakes.\nBut I'm here to mend the heart I didn't break.\n\n"
                    "[chorus]\nEvery road you take, I'll be one step behind.\nEvery dream you chase, I'm reaching for the light.\nYou can't fight this feeling now.\nI won't back down.\nYou know you can't deny it now.\n I won't back down \n\n"
                    "[verse]\nThey might say I'm foolish, chasing after you.\nBut they don't feel this love the way we do.\nMy heart beats only for you, can't you see?\nI won't let you slip away from me. \n\n"
                    "[chorus]\nEvery road you take, I'll be one step behind.\nEvery dream you chase, I'm reaching for the light.\nYou can't fight this feeling now.\nI won't back down.\nYou know you can't deny it now.\n I won't back down \n\n"
                    "[bridge]\nNo, I won't back down, won't turn around.\nUntil you're back where you belong.\nI'll cross the oceans wide, stand by your side.\nTogether we are strong. \n\n"
                    "[outro]\nEvery road you take, I'll be one step behind.\nEvery dream you chase, love's the tie that binds.\nYou can't fight this feeling now.\nI won't back down.", "multiline": True, "tooltip": "歌词内容，使用[verse]、[chorus]、[bridge]等标签分段，音乐时长将根据歌词段落数量自动确定"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED, "tooltip": "随机种子，用于控制生成结果的随机性，相同种子产生相同结果"}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 2.0, "step": 0.1, "tooltip": "重复惩罚系数，值越大越能避免重复内容，建议范围1.0-1.5"}),
                "prompt_start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 300.0, "step": 0.5, "tooltip": "音频提示的开始时间（仅在使用音频提示时有效）"}),
                "prompt_end_time": ("FLOAT", {"default": 6.0, "min": 1.0, "max": 300.0, "step": 0.5, "tooltip": "音频提示的结束时间（仅在使用音频提示时有效）"}),
                "max_new_tokens": ("INT", {"default": 8192, "min": 2944, "max": 15000, "step": 1, "display": "number", "tooltip": "最大生成token数量，影响生成质量和速度。注意：过大的值可能导致内存不足或生成失败"}),
                "use_dual_tracks_prompt":("BOOLEAN",{"default":True, "tooltip": "是否使用双轨音频提示（人声+伴奏分离）"}),
                "use_audio_prompt":("BOOLEAN",{"default":False, "tooltip": "是否使用单轨音频提示作为参考"}),
                "offload_model":("BOOLEAN",{"default":True, "tooltip": "是否在生成完成后卸载模型以释放显存"}),
                "stage1_no_guidance":("BOOLEAN",{"default":True, "tooltip": "是否禁用第一阶段的引导生成，禁用可能提高生成速度"}),
            },
            "optional": {
                "manual_segments": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1, "display": "number", "tooltip": "限制使用的歌词段落数量，0表示使用所有歌词段落"}),
                "custom_audio_prompt": ("AUDIO", {"tooltip": "自定义音频参考文件（可选），如果不提供则使用默认的pop.00001.mp3"}),
                "custom_vocal_track": ("AUDIO", {"tooltip": "自定义人声轨道文件（可选），用于双轨模式，如果不提供则使用默认的pop.00001.Vocals.mp3"}),
                "custom_instrumental_track": ("AUDIO", {"tooltip": "自定义伴奏轨道文件（可选），用于双轨模式，如果不提供则使用默认的pop.00001.Instrumental.mp3"}),
            }
        }

    RETURN_TYPES = ("STAGE_SET", "quantization_model",)
    RETURN_NAMES = ("stage1_set","info")
    FUNCTION = "sampler_main"
    CATEGORY = "YUE"

    def sampler_main(self, model, genres_prompt, lyrics_prompt, seed, repetition_penalty, prompt_start_time, prompt_end_time, max_new_tokens,
                     use_dual_tracks_prompt, use_audio_prompt, offload_model, stage1_no_guidance, manual_segments=0, 
                     custom_audio_prompt=None, custom_vocal_track=None, custom_instrumental_track=None):
        
        # 使用自定义音频文件或默认文件
        if custom_instrumental_track is not None:
            instrumental_track_prompt_path = custom_instrumental_track
        else:
            instrumental_track_prompt_path = os.path.join(current_node_path, "prompt_egs/pop.00001.Instrumental.mp3")
            
        if custom_vocal_track is not None:
            vocal_track_prompt_path = custom_vocal_track
        else:
            vocal_track_prompt_path = os.path.join(current_node_path, "prompt_egs/pop.00001.Vocals.mp3")
            
        if custom_audio_prompt is not None:
            audio_prompt_path = custom_audio_prompt
        else:
            audio_prompt_path = os.path.join(current_node_path, "prompt_egs/pop.00001.mp3")

        stage1_output_dir=os.path.join(folder_paths.get_output_directory(), "stage1")
        os.makedirs(stage1_output_dir, exist_ok=True)

        #genre_txt_path=os.path.join(current_node_path, "prompt_egs/genre.txt")
        #lyrics_txt_path=os.path.join(current_node_path, "prompt_egs/lyrics.txt")
        
        quantization_model=model.get("quantization_model")
        mmtokenizer=model.get("mmtokenizer")
        codec_model=model.get("codec_model")
        codectool=model.get("codectool")
        codectool_stage2=model.get("codectool_stage2")
        mmgp_profile=model.get("mmgp_profile")
        
        if quantization_model=="exllamav2":
            from .inference.infer_stage1 import SampleSettings
            # with open(lyrics_txt_path) as f:
            #     lyrics = f.read().strip()    
            lyrics=lyrics_prompt.strip()   
            pipeline=model.get("stage_1_model")
            
            # Calculate run_n_segments based on manual_segments parameter
            lyrics_segments = split_lyrics(lyrics_prompt.strip())
            if manual_segments > 0:
                run_n_segments = min(manual_segments, len(lyrics_segments))
                print(f"Using manual segments limit: {run_n_segments} (max: {manual_segments}, available lyrics: {len(lyrics_segments)})")
            else:
                run_n_segments = len(lyrics_segments)
                print(f"Using all lyrics segments: {run_n_segments}")
            
            seed_everything(seed)
            raw_output = pipeline.generate(
                use_dual_tracks_prompt=use_dual_tracks_prompt,
                vocal_track_prompt_path=vocal_track_prompt_path,
                instrumental_track_prompt_path=instrumental_track_prompt_path,
                use_audio_prompt=use_audio_prompt,
                audio_prompt_path=audio_prompt_path,
                genres=genres_prompt.strip(),
                lyrics=lyrics,
                run_n_segments=run_n_segments,
                max_new_tokens=max_new_tokens,
                prompt_start_time=prompt_start_time,
                prompt_end_time=prompt_end_time,
                sample_settings=SampleSettings(use_guidance=not stage1_no_guidance, repetition_penalty=repetition_penalty),
            )

            #print(raw_output)

            # Save result
            pipeline.save(raw_output, folder_paths.get_output_directory(), use_audio_prompt, use_dual_tracks_prompt)
            stage1_output_set=None
        else:

            # with open(genre_txt_path) as f:
            #     genres_prompt = f.read().strip()
            # with open(lyrics_txt_path) as f:
            #     lyrics_prompt = split_lyrics(f.read())
            
            # Debug: Print original lyrics input
            print(f"DEBUG: Original lyrics input length: {len(lyrics_prompt.strip())}")
            print(f"DEBUG: Original lyrics preview: {lyrics_prompt.strip()[:200]}...")
            
            lyrics=split_lyrics(lyrics_prompt.strip()) 
            
            # Check if lyrics is empty
            if not lyrics or len(lyrics) == 0:
                raise ValueError(f"No lyrics segments found after splitting. Original input: {lyrics_prompt.strip()[:200]}...")
            
            stage_1_model=model.get("stage_1_model")
            
    
            seed_everything(seed)
            # Call the function and print the result
            stage1_output_set = []
            # Tips:
            # genre tags support instrumental，genre，mood，vocal timbr and vocal gender
            # all kinds of tags are needed   
            # intruction
            full_lyrics = "\n".join(lyrics)
            prompt_texts = [f"Generate music from the given lyrics segment by segment.\n[Genre] {genres_prompt}\n{full_lyrics}"]
            prompt_texts += lyrics

            random_id = uuid.uuid4()
            output_seq = None
            # Here is suggested decoding config
            top_p = 0.93
            temperature = 1.0
            #repetition_penalty = args.repetition_penalty
            # special tokens
            start_of_segment = mmtokenizer.tokenize('[start_of_segment]')
            end_of_segment = mmtokenizer.tokenize('[end_of_segment]')
            
            # Calculate run_n_segments based on lyrics or manual_segments
            if manual_segments > 0:
                # Use manual setting as a limit
                run_n_segments = min(manual_segments, len(lyrics))
                print(f"Using manual segments limit: {run_n_segments} (max: {manual_segments}, available lyrics: {len(lyrics)})")
            else:
                # Use all available lyrics segments
                run_n_segments = len(lyrics)
                print(f"Using all lyrics segments: {run_n_segments}")
            
            # Additional safety checks
            if len(lyrics) == 0:
                raise ValueError("No lyrics provided. Cannot generate music without lyrics.")
            
            if run_n_segments <= 1:
                print(f"Warning: run_n_segments is {run_n_segments}, which is too small for proper generation.")
                if len(lyrics) >= 2:
                    run_n_segments = 2
                    print(f"Setting to minimum of 2 segments.")
                else:
                    run_n_segments = len(lyrics)
                    print(f"Using all available lyrics segments: {run_n_segments}")
            
            # Format text prompt
            #run_n_segments = min(run_n_segment+1, len(lyrics_prompt))
            #run_n_segments = min(run_n_segment, len(lyrics_prompt))
            
            # Initialize raw_output to avoid UnboundLocalError
            raw_output = None
            first_valid_iteration = True
            
            for i, p in enumerate(tqdm(prompt_texts[:run_n_segments], desc="Stage1 inference...")):
                section_text = p.replace('[start_of_segment]', '').replace('[end_of_segment]', '')
                guidance_scale = 1.5 if i <=1 else 1.2
                
                if i==0:
                    continue
                
                # Prepare prompt_ids based on iteration
                print(f"DEBUG: Preparing prompt_ids for segment {i}")
                if i==1:
                    print(f"DEBUG: Processing first segment (i==1)")
                    if use_dual_tracks_prompt or use_audio_prompt:
                        print(f"DEBUG: Using audio prompt (dual_tracks: {use_dual_tracks_prompt}, audio: {use_audio_prompt})")
                        if use_dual_tracks_prompt:
                            vocals_ids = load_audio_mono(vocal_track_prompt_path)
                            instrumental_ids = load_audio_mono(instrumental_track_prompt_path)
                            vocals_ids = encode_audio(codec_model, vocals_ids, device, target_bw=0.5)
                            instrumental_ids = encode_audio(codec_model, instrumental_ids, device, target_bw=0.5)
                            vocals_ids = codectool.npy2ids(vocals_ids[0])
                            instrumental_ids = codectool.npy2ids(instrumental_ids[0])
                            ids_segment_interleaved = rearrange([np.array(vocals_ids), np.array(instrumental_ids)], 'b n -> (n b)')
                            audio_prompt_codec = ids_segment_interleaved[int(prompt_start_time*50*2): int(prompt_end_time*50*2)]
                            audio_prompt_codec = audio_prompt_codec.tolist()
                        elif use_audio_prompt:
                            audio_prompt = load_audio_mono(audio_prompt_path)
                            raw_codes = encode_audio(codec_model, audio_prompt, device, target_bw=0.5)
                            # Format audio prompt
                            code_ids = codectool.npy2ids(raw_codes[0])
                            audio_prompt_codec = code_ids[int(prompt_start_time *50): int(prompt_end_time *50)] # 50 is tps of xcodec
                        audio_prompt_codec_ids = [mmtokenizer.soa] + codectool.sep_ids + audio_prompt_codec + [mmtokenizer.eoa]
                        sentence_ids = mmtokenizer.tokenize("[start_of_reference]") +  audio_prompt_codec_ids + mmtokenizer.tokenize("[end_of_reference]")
                        head_id = mmtokenizer.tokenize(prompt_texts[0]) + sentence_ids
                    else:
                        head_id = mmtokenizer.tokenize(prompt_texts[0])
                    
                    tokenized_section = mmtokenizer.tokenize(section_text)
                    prompt_ids = head_id + start_of_segment + tokenized_section + [mmtokenizer.soa] + codectool.sep_ids
                else:
                    tokenized_section = mmtokenizer.tokenize(section_text)
                    prompt_ids = end_of_segment + start_of_segment + tokenized_section + [mmtokenizer.soa] + codectool.sep_ids

                # Validate prompt_ids is not empty
                if not prompt_ids:
                    continue
                
                prompt_ids = torch.as_tensor(prompt_ids).unsqueeze(0).to(device)
                
                # Validate prompt_ids tensor
                if prompt_ids.numel() == 0:
                    continue
                
                # Prepare input_ids
                print(f"DEBUG: Preparing input_ids, first_valid_iteration: {first_valid_iteration}")
                if not first_valid_iteration and raw_output is not None:
                    print(f"DEBUG: Concatenating with existing raw_output (shape: {raw_output.shape})")
                    input_ids = torch.cat([raw_output, prompt_ids], dim=1)
                else:
                    print(f"DEBUG: Using prompt_ids as input_ids")
                    input_ids = prompt_ids
                
                print(f"DEBUG: input_ids shape: {input_ids.shape}")
                
                # Validate input_ids
                if input_ids.numel() == 0:
                    print(f"ERROR: input_ids tensor is empty for section {i}")
                    print(f"DEBUG: Skipping section {i}")
                    continue
                
                # Use window slicing in case output sequence exceeds the context of model
                # Ensure max_new_tokens doesn't exceed model context limit
                model_context_limit = 16384
                effective_max_new_tokens = min(max_new_tokens, model_context_limit - 1000)  # Reserve 1000 tokens for input
                max_context = model_context_limit - effective_max_new_tokens - 1
                
                if effective_max_new_tokens != max_new_tokens:
                    print(f"WARNING: max_new_tokens ({max_new_tokens}) too large, using {effective_max_new_tokens} instead")
                
                if input_ids.shape[-1] > max_context:
                    print(f'Section {i}: output length {input_ids.shape[-1]} exceeding context length {max_context}, now using the last {max_context} tokens.')
                    input_ids = input_ids[:, -(max_context):]
                
                # Final validation before generation
                if input_ids.shape[-1] == 0:
                    continue
                
                try:
                    with torch.no_grad():
                        output_seq = stage_1_model.generate(
                            input_ids=input_ids, 
                            max_new_tokens=effective_max_new_tokens, 
                            min_new_tokens=100, 
                            do_sample=True, 
                            top_p=top_p,
                            temperature=temperature, 
                            repetition_penalty=repetition_penalty, 
                            eos_token_id=mmtokenizer.eoa,
                            pad_token_id=mmtokenizer.eoa,
                            logits_processor=LogitsProcessorList([BlockTokenRangeProcessor(0, 32002), BlockTokenRangeProcessor(32016, 32016)]),
                            guidance_scale=guidance_scale,
                            )
                        
                        if output_seq[0][-1].item() != mmtokenizer.eoa:
                            tensor_eoa = torch.as_tensor([[mmtokenizer.eoa]]).to(stage_1_model.device)
                            output_seq = torch.cat((output_seq, tensor_eoa), dim=1)
                            
                except Exception as e:
                    print(f"ERROR: Generation failed for section {i}: {str(e)}")
                    continue
                
                # Update raw_output - use first_valid_iteration flag instead of hardcoded i==1
                if first_valid_iteration:
                    raw_output = output_seq
                    first_valid_iteration = False
                    print(f"Initialized raw_output at section {i} with shape: {raw_output.shape}")
                else:
                    raw_output = torch.cat([raw_output, prompt_ids, output_seq[:, input_ids.shape[-1]:]], dim=1)
                    print(f"Updated raw_output at section {i}, new shape: {raw_output.shape}")

            # Check if raw_output was properly initialized
            if raw_output is None:
                raise ValueError("Generation failed: raw_output was not initialized. Please check your lyrics format and try again.")

            # save raw output and check sanity
            ids = raw_output[0].cpu().numpy()
            soa_idx = np.where(ids == mmtokenizer.soa)[0].tolist()
            eoa_idx = np.where(ids == mmtokenizer.eoa)[0].tolist()
            if len(soa_idx)!=len(eoa_idx):
                raise ValueError(f'invalid pairs of soa and eoa, Num of soa: {len(soa_idx)}, Num of eoa: {len(eoa_idx)}')

            vocals = []
            instrumentals = []
            range_begin = 1 if use_audio_prompt or use_dual_tracks_prompt else 0
            for i in range(range_begin, len(soa_idx)):
                codec_ids = ids[soa_idx[i]+1:eoa_idx[i]]
                if codec_ids[0] == 32016:
                    codec_ids = codec_ids[1:]
                codec_ids = codec_ids[:2 * (codec_ids.shape[0] // 2)]
                vocals_ids = codectool.ids2npy(rearrange(codec_ids,"(n b) -> b n", b=2)[0])
                vocals.append(vocals_ids)
                instrumentals_ids = codectool.ids2npy(rearrange(codec_ids,"(n b) -> b n", b=2)[1])
                instrumentals.append(instrumentals_ids)
            vocals = np.concatenate(vocals, axis=1)
            instrumentals = np.concatenate(instrumentals, axis=1)
            vocal_save_path = os.path.join(stage1_output_dir, f"{genres_prompt.replace(' ', '-')}_tp{top_p}_T{temperature}_rp{repetition_penalty}_maxtk{max_new_tokens}_{random_id}_vtrack".replace('.', '@')+'.npy')
            inst_save_path = os.path.join(stage1_output_dir, f"{genres_prompt.replace(' ', '-')}_tp{top_p}_T{temperature}_rp{repetition_penalty}_maxtk{max_new_tokens}_{random_id}_itrack".replace('.', '@')+'.npy')
            np.save(vocal_save_path, vocals)
            np.save(inst_save_path, instrumentals)
            stage1_output_set.append(vocal_save_path)
            stage1_output_set.append(inst_save_path)

            # offload model
            if offload_model:
                stage_1_model.cpu()
                stage_1_model=None            
                torch.cuda.empty_cache()
                gc.collect()            
                
        torch.cuda.empty_cache()
        gc.collect()  
        return ({"stage1_output_set":stage1_output_set,"codec_model":codec_model,"codectool_stage2":codectool_stage2,"mmtokenizer":mmtokenizer,"codectool":codectool,"quantization_model":quantization_model,},{"quantization_model":quantization_model,"mmgp_profile":mmgp_profile,},)

class YUE_Stage_B_Loader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "info": ("quantization_model", {"tooltip": "从Stage A传递的量化模型信息"}),
                "stage_B_repo": ("STRING",{"default": "YuE-s2-1B-general", "tooltip": "Stage B模型仓库名称，用于音乐生成的第二阶段(音频合成)"}),
                "stage2_cache_size": ("INT",{"default": 262144, "min": 4096, "max": MAX_SEED, "step": 64, "display": "number", "tooltip": "Stage2缓存大小，控制音频合成阶段的缓存。更大的值可显著提升合成速度"}),
                "stage2_batch_size": ("INT",{"default": 2, "min": 1, "max": 64, "step": 1, "display": "number", "tooltip": "Stage2批处理大小，影响音频合成的并行度。增大可提升速度但占用更多显存"}),
                "exllamav2_cache_mode": (["FP16","Q8","Q6", "Q4"], {"tooltip": "ExLlamaV2缓存模式：FP16(最高质量)、Q8(8位量化)、Q6(6位量化)、Q4(4位量化，最省内存)"}),
                "use_mmgp":("BOOLEAN",{"default":True, "tooltip": "是否使用MMGP(多模态生成预处理)，建议开启以获得更好的音频合成质量"}),
            },
        }

    RETURN_TYPES = ("MODEL_YUE_B",)
    RETURN_NAMES = ("model",)
    FUNCTION = "loader_main"
    CATEGORY = "YUE"

    def loader_main(self,info,stage_B_repo,stage2_cache_size,stage2_batch_size,exllamav2_cache_mode,use_mmgp):
        # Debug: Print original stage_B_repo value
        print(f"[DEBUG] Original stage_B_repo: {stage_B_repo}")
        print(f"[DEBUG] YUE_weigths_path: {YUE_weigths_path}")
        
        # Convert stage_B_repo to full local path
        original_stage_B_repo = stage_B_repo
        local_model_path = os.path.join(YUE_weigths_path, stage_B_repo)
        
        # Debug: Print path construction
        print(f"[DEBUG] Constructed local path: {local_model_path}")
        print(f"[DEBUG] Local path exists: {os.path.exists(local_model_path)}")
        
        # Check if local model folder exists
        if os.path.exists(local_model_path) and os.path.isdir(local_model_path):
            stage_B_repo = local_model_path
            print(f"[DEBUG] Using local model path: {stage_B_repo}")
        else:
            print(f"[ERROR] Local model folder '{local_model_path}' not found")
            print(f"[DEBUG] Available files in yue folder: {folder_paths.get_filename_list('yue')}")
            # Fallback to original value for Hugging Face download
            stage_B_repo = f"m-a-p/{original_stage_B_repo}"
            print(f"[DEBUG] Falling back to Hugging Face repo: {stage_B_repo}")

        quantization_model=info.get("quantization_model")
        mmgp_profile=info.get("mmgp_profile")
       
        if not use_mmgp:
            if quantization_model=="exllamav2":
                from .inference.infer_stage2 import Stage2Pipeline_EXL2 ,Stage2Pipeline_HF
                if exllamav2_cache_mode=="FP16":
                    print("**********Loading fp16*********")
                    model_stage2=Stage2Pipeline_HF(model_path=stage_B_repo, device=device, batch_size=stage2_batch_size)
                else:
                    print("**********Loading exllamav2*********")
                    model_stage2=Stage2Pipeline_EXL2(model_path=stage_B_repo, device=device, cache_size=stage2_cache_size, cache_mode=exllamav2_cache_mode)
            else:
                if quantization_model=="int8":
                    print("**********Loading int8*********")
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                    model_stage2 = AutoModelForCausalLM.from_pretrained(
                        stage_B_repo, 
                        torch_dtype=torch.bfloat16,
                        quantization_config=quantization_config,
                        attn_implementation="flash_attention_2",
                        #device_map="auto"
                        )
                elif quantization_model=="int4":
                    print("**********Loading int4*********")
                    quantization_config = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type="int4"
                                )
                    model_stage2 = AutoModelForCausalLM.from_pretrained(
                        stage_B_repo, 
                        torch_dtype=torch.bfloat16,
                        quantization_config=quantization_config,
                        attn_implementation="flash_attention_2",)       
                else:
                    print("**********Loading fp16*********")
                    model_stage2 = AutoModelForCausalLM.from_pretrained(
                        stage_B_repo, 
                        torch_dtype=torch.bfloat16,
                        attn_implementation="flash_attention_2",
                        # device_map="auto",
                        )
                if torch.__version__ >= "2.0.0":
                    model_stage2 = torch.compile(model_stage2)

                if quantization_model=="fp16":
                    model_stage2 = model_stage2.to('cuda:0')
                model_stage2.eval()
        else:
            print("**********Loading mmpg*********")
            from mmgp import offload
            model_stage2 = AutoModelForCausalLM.from_pretrained(
                    stage_B_repo, 
                    torch_dtype=torch.float16,
                    attn_implementation="flash_attention_2"
                    )
            if torch.__version__ >= "2.0.0":
                    model_stage2 = torch.compile(model_stage2)
            model_stage2.to("cpu")
            model_stage2.eval()
            compile= True
            pipe = { "stage2": model_stage2,}
            kwargs  = {}
            if mmgp_profile == 4 :
                kwargs["budgets"] =  { "transformer": 3000, "*" : 5000 }
            elif mmgp_profile == 2:
                kwargs["budgets"] =  5000

            quantizeTransformer = mmgp_profile == 3 or mmgp_profile == 4 or mmgp_profile == 5 
            offload.profile(pipe, profile_no = mmgp_profile,  compile = compile, quantizeTransformer= quantizeTransformer,  verboseLevel= 1, **kwargs )
        gc.collect()
        torch.cuda.empty_cache()
        return ({"model_stage2":model_stage2,"stage2_batch_size":stage2_batch_size},)


class YUE_Stage_B_Sampler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        ckpt_list_vocal = [i for i in folder_paths.get_filename_list("yue") if "131" in i]
        ckpt_list_inst = [i for i in folder_paths.get_filename_list("yue") if "151" in i]
        return {
            "required": {
                "stage1_set": ("STAGE_SET",),
                "model": ("MODEL_YUE_B",),
                "vocal_decoder_ckpt": (["none"] + ckpt_list_vocal,),
                "inst_decoder_ckpt": (["none"] + ckpt_list_inst,),
                "rescale":("BOOLEAN",{"default":True}),
            }}

    RETURN_TYPES = ("AUDIO","STRING", )
    RETURN_NAMES = ("audio","string",)
    FUNCTION = "sampler_main"
    CATEGORY = "YUE"

    def sampler_main(self,stage1_set, model,vocal_decoder_ckpt, inst_decoder_ckpt,rescale):
        file_prefix = ''.join(random.choice("0123456789") for _ in range(6))
           
        quantization_model=stage1_set.get("quantization_model")
        model_stage2=model.get("model_stage2")
        stage2_batch_size=model.get("stage2_batch_size")
        config_path=os.path.join(current_node_path,'inference/xcodec_mini_infer/decoders/config.yaml')
        vocal_decoder_path=folder_paths.get_full_path("yue", vocal_decoder_ckpt)
        inst_decoder_path=folder_paths.get_full_path("yue", inst_decoder_ckpt)
        output_dir=folder_paths.get_output_directory()

        if quantization_model=="exllamav2":
            from .inference.infer_postprocess import post_process
            outputs = model_stage2.generate(output_dir=output_dir)
            model_stage2.save(output_dir=output_dir, outputs=outputs)
            codec_model=stage1_set.get("codec_model")
            mix_output,c_file=post_process(codec_model, device, output_dir, config_path, vocal_decoder_path, inst_decoder_path, rescale,file_prefix)
        else:
            
            stage1_output_set=stage1_set.get("stage1_output_set")
            codectool_stage2=stage1_set.get("codectool_stage2")
            codectool=stage1_set.get("codectool")
            codec_model=stage1_set.get("codec_model")
            mmtokenizer=stage1_set.get("mmtokenizer")

            stage2_output_dir=os.path.join(folder_paths.get_output_directory(), "stage2")
            os.makedirs(stage2_output_dir, exist_ok=True)

            stage2_result = stage2_inference(model_stage2, stage1_output_set, stage2_output_dir,codectool_stage2,mmtokenizer,codectool,device, batch_size=stage2_batch_size)
            print(stage2_result)
            print('Stage 2 DONE.\n')
            # reconstruct tracks
            recons_output_dir = os.path.join(output_dir, "recons")
            recons_mix_dir = os.path.join(recons_output_dir, 'mix')
            os.makedirs(recons_mix_dir, exist_ok=True)
            tracks = []
            for npy in stage2_result:
                codec_result = np.load(npy)
                decodec_rlt=[]
                with torch.no_grad():
                    decoded_waveform = codec_model.decode(torch.as_tensor(codec_result.astype(np.int16), dtype=torch.long).unsqueeze(0).permute(1, 0, 2).to(device))
                decoded_waveform = decoded_waveform.cpu().squeeze(0)
                decodec_rlt.append(torch.as_tensor(decoded_waveform))
                decodec_rlt = torch.cat(decodec_rlt, dim=-1)
                save_path = os.path.join(recons_output_dir, os.path.splitext(os.path.basename(npy))[0] + ".mp3")
                tracks.append(save_path)
                save_audio(decodec_rlt, save_path, 16000)
            # mix tracks
            for inst_path in tracks:
                try:
                    if (inst_path.endswith('.wav') or inst_path.endswith('.mp3')) \
                        and '_itrack' in inst_path:
                        # find pair
                        vocal_path = inst_path.replace('_itrack', '_vtrack')
                        if not os.path.exists(vocal_path):
                            continue
                        # mix
                        recons_mix = os.path.join(recons_mix_dir, os.path.basename(inst_path).replace('_itrack', '_mixed'))
                        vocal_stem, sr = sf.read(inst_path)
                        instrumental_stem, _ = sf.read(vocal_path)
                        mix_stem = (vocal_stem + instrumental_stem) / 1
                        sf.write(recons_mix, mix_stem, sr)
                except Exception as e:
                    print(e)

            # vocoder to upsample audios
            vocal_decoder, inst_decoder = build_codec_model(config_path, vocal_decoder_path, inst_decoder_path)
            vocoder_output_dir = os.path.join(output_dir, 'vocoder')
            vocoder_stems_dir = os.path.join(vocoder_output_dir, 'stems')
            vocoder_mix_dir = os.path.join(vocoder_output_dir, 'mix')
            os.makedirs(vocoder_mix_dir, exist_ok=True)
            os.makedirs(vocoder_stems_dir, exist_ok=True)
            args={}
            for npy in stage2_result:
                if '_itrack' in npy:
                    # Process instrumental
                    instrumental_output = process_audio(
                        npy,
                        os.path.join(vocoder_stems_dir, 'itrack.mp3'),
                        rescale,
                        args,
                        inst_decoder,
                        codec_model
                    )
                else:
                    # Process vocal
                    vocal_output = process_audio(
                        npy,
                        os.path.join(vocoder_stems_dir, 'vtrack.mp3'),
                        rescale,
                        args,
                        vocal_decoder,
                        codec_model
                    )
            # mix tracks
            try:
                mix_output = instrumental_output + vocal_output
                vocoder_mix = os.path.join(vocoder_mix_dir, os.path.basename(recons_mix))
                save_audio(mix_output, vocoder_mix, 44100, rescale)
                print(f"Created mix: {vocoder_mix}")
            except RuntimeError as e:
                print(e)
                print(f"mix {vocoder_mix} failed! inst: {instrumental_output.shape}, vocal: {vocal_output.shape}")
            c_file=os.path.join(output_dir, f"yue_{file_prefix}_{os.path.basename(recons_mix)}")
            # Post process
            replace_low_freq_with_energy_matched(
                a_file=recons_mix,     # 16kHz
                b_file=vocoder_mix,     # 48kHz
                c_file=c_file,
                cutoff_freq=5500.0
            )
        print(mix_output.shape)
        
        audio= {"waveform": mix_output.unsqueeze(0), "sample_rate": 44100}
        return(audio,c_file,)

NODE_CLASS_MAPPINGS = {
    "YUE_Stage_A_Loader": YUE_Stage_A_Loader,
    "YUE_Stage_A_Sampler": YUE_Stage_A_Sampler,
    "YUE_Stage_B_Loader": YUE_Stage_B_Loader,
    "YUE_Stage_B_Sampler": YUE_Stage_B_Sampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YUE_Stage_A_Loader": "YUE_Stage_A_Loader",
    "YUE_Stage_A_Sampler": "YUE_Stage_A_Sampler",
    "YUE_Stage_B_Loader": "YUE_Stage_B_Loader",
    "YUE_Stage_B_Sampler": "YUE_Stage_B_Sampler",
}




