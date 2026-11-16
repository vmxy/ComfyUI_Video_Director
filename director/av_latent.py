import torch
import torchaudio
from comfy_extras.nodes_lt import LTXVSeparateAVLatent, LTXVConcatAVLatent


def replace_audio_and_lock(av_latent, audio_vae, audio):
    """
    1. 从 H3/LTX 的 AV latent 中分离出 video_latent 和旧 audio_latent
    2. 用 audio_vae 将 new_audio 编码为新的 audio latent，并构建全零 noise_mask（锁定音频）
    3. 将新 audio_latent 与 video_latent 通过内置 AV 拼接逻辑重组为新 latent
    
    参数:
        av_latent: dict, {"samples": NestedTensor(video, audio), ...}
        audio_vae: VAE 对象，H3 专用音频 VAE
        audio: dict, {"waveform": Tensor[B,C,L], "sample_rate": int}
    返回:
        dict, 新的 AV latent，其中 audio 流被替换且 noise_mask 锁定为 0
    """
    # ========== 第 1 步：分离 AV latent ==========
    # 复用 ComfyUI 内置 LTXVSeparateAVLatent 的逻辑
    # 其 execute 返回 (video_latent, audio_latent)，内部调用 NestedTensor.unbind()
    video_latent, audio_latent = LTXVSeparateAVLatent.execute(av_latent)

     # ========== 第 2 步：用音频 VAE 编码新音频 ==========
    waveform = audio["waveform"]
    sr = audio["sample_rate"]
    
    # H3 音频 VAE 的采样率通常为 32000 Hz，而非通用 VAEEncodeAudio 的 44100
    vae_sr = getattr(audio_vae, "audio_sample_rate", 44100)
    if sr != vae_sr:
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    
    # 编码：movedim(1, -1) 将 [B,C,L] 转为 [B,L,C]
    audio_samples = audio_vae.encode(waveform.movedim(1, -1))
    # 编码结果形状应为 [B, 32, 2, T_audio]，H3 的音频 latent 契约
    
    # 构建新的 audio_latent 字典
    new_audio_latent = {"samples": audio_samples}
    
    # ========== 构建纯块遮罩（透明为 0）==========
    # H3 采样器将 noise_mask 为 0 的区域视为“保留原值，不重绘”
    # 因此音频流全零遮罩 = 锁定音频，不让模型重新生成
    audio_mask = torch.zeros_like(audio_samples)
    new_audio_latent["noise_mask"] = audio_mask

    # ========== 第 3 步：AV 拼接 ==========
    # 复用 ComfyUI 内置 LTXVConcatAVLatent 的逻辑
    # 注意：该节点会自动处理音频时长对齐（截断/补零），并合并 noise_mask
    node_output = LTXVConcatAVLatent.execute(video_latent, new_audio_latent)
    # NodeOutput 的位置参数就是输出值，取第一个
    result_latent = node_output[0]   # 或 node_output.args[0]
    
    # 如果 video_latent 原本有 noise_mask，拼接后的 latent 会包含合并后的遮罩
    # 其中 video 部分沿用原 mask，audio 部分为全零（锁定）
    return result_latent
