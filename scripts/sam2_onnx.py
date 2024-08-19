from sam2.modeling.sam2_base import SAM2Base, NO_OBJ_SCORE
import torch


class SAM2Onnx(SAM2Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self,
                frame_idx,
                is_init_cond_frame,
                current_vision_feats,
                current_vision_pos_embeds,
                feat_sizes,
                point_inputs,
                mask_inputs,
                output_dict,
                num_frames,
                track_in_reverse,
                run_mem_encoder,
                prev_sam_mask_logits,):
        return
        self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=num_frames,
            track_in_reverse=track_in_reverse,
            run_mem_encoder=run_mem_encoder,
            prev_sam_mask_logits=prev_sam_mask_logits,)
