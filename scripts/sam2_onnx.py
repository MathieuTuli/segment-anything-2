from sam2.modeling.sam2_base import SAM2Base, NO_OBJ_SCORE


class SAM2Onnx(SAM2Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
