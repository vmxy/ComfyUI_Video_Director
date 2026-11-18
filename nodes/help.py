_CATEGORY = "MiniMaxH3"


# 计算分片帧数
class VideoCalSegFrame:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "second": (
                    "FLOAT", 
                    {
                        "default": 10, "min": 1, "step": 1, 
                        "tooltip": "时长(秒)"
                    }),
                "fps": (
                    "FLOAT", 
                    {
                        "default": 24, "min": 16, "step": 1, 
                        "tooltip": "帧率"
                    }),
                "motion_frame": (
                    "INT", 
                    {
                        "default": 0, "min": 0, "step": 1, 
                        "tooltip": "监听多少侦",
                    }),
            },
            "optional": { # 放入 optional
               
            }
        }
    
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("sample_frame",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    
    def execute(self, second: float = 10, fps: float = 24.0, motion_frame: int = 0):
        #_duration_to_minimax_frames
        from ..director.fl2v_timeline import _duration_to_minimax_frames
        fc = _duration_to_minimax_frames(second, fps)
        from ..director.frame_align import minimax_align_frame_count
        sample_frame = minimax_align_frame_count(fc + motion_frame)
        #minimax_align_frame_count(visible + ctx)
        return (sample_frame, )



def _cal_frame_count(second: float = 10, fps: float = 24.0, motion_frame: int = 0, keep_tail: bool = False):
    from ..director.fl2v_timeline import _duration_to_minimax_frames
    fc = _duration_to_minimax_frames(second, fps)
    from ..director.frame_align import minimax_align_frame_count
    sample_frame = minimax_align_frame_count(fc + motion_frame)
    return (sample_frame, sample_frame - motion_frame if keep_tail else fc)


class SliceAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ref_audio": (
                    "AUDIO", 
                    {
                        "forceInput": True,
                        "tooltip": "输入音频(可选)"
                    }),
                "start_second": (
                    "FLOAT", 
                    {
                        "default": 0, "min": 0, "step": 1, 
                        "tooltip": "开始(秒)"
                    }),
                "duration": (
                    "FLOAT", 
                    {
                        "default": 10.0, "min": 3.0, "step": 1.0, 
                        "tooltip": "时长(秒)"
                    }),
                "fps": (
                    "FLOAT", 
                    {
                        "default": 24.0, "min": 16.0, "step": 1.0, 
                        "tooltip": "侦率"
                    }),
                "motion_frame": (
                    "INT", 
                    {
                        "default": 22, "min": 0, "step": 1, 
                        "tooltip": "监听多少侦"
                    }),
                "keep_tail": (
                    "BOOLEAN", 
                    {
                        "default": False, 
                        "tooltip": "保留尾部完整性"
                    }),
            },
        }
    
    RETURN_TYPES = ("AUDIO", "AUDIO", "FLOAT", "FLOAT", "FLOAT", "INT", "BOOLEAN", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("ref_audio", "slice_audio", "next_start_second", "duration", "fps", "motion_frame", "keep_tail",
        "motion_second", "start_second", "duration_real"
    )
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    
    def execute(self, ref_audio, start_second, duration, fps, motion_frame, keep_tail: bool = False):
        # 总共要采样多少帧
        sample_frame, visiable_frame = _cal_frame_count(second=duration, fps=fps, motion_frame=motion_frame, keep_tail=keep_tail)
        motion_second = round(motion_frame / fps, 3) #监听上文多少秒
        visiable_second = round(visiable_frame / fps, 3) #扩展多少秒
        duration_real = round(sample_frame / fps, 3)
        start_second_real = 0 if start_second <= motion_second else (start_second - motion_second) 
        next_start_second = start_second + visiable_second
        print(f"--- sample_frame={sample_frame} visiable_frame={visiable_frame} visiable_second={visiable_second} start_second_real={start_second_real} start_second={start_second} next_start_second={next_start_second} motion_second={motion_second} duration_real={duration_real}")
        from comfy_extras.nodes_audio import TrimAudioDuration
        # 假设 ref_audio 是你已有的 AUDIO 字典，例如 {"waveform": tensor, "sample_rate": 48000}
        # start_time 单位秒，duration 单位秒
        out_audio = TrimAudioDuration.execute(ref_audio, start_index=start_second_real, duration=duration_real)[0]
        return (ref_audio, out_audio, next_start_second, duration, fps, motion_frame, keep_tail,
             motion_second, start_second_real, duration_real, 
        )
    
