echo "Converting all models..."

echo "Converting SAM2-Hiera-Tiny models..."
python scripts/export_sam2.py --checkpoint checkpoints/sam2_hiera_tiny.pt \
    --output_encoder checkpoints/onnx/sam2_hiera_tiny.encoder.onnx \
    --output_decoder checkpoints/onnx/sam2_hiera_tiny.decoder.onnx \
    --model_type sam2_hiera_tiny

echo "Converting SAM2-Hiera-Small models..."
python scripts/export_sam2.py --checkpoint checkpoints/sam2_hiera_small.pt \
    --output_encoder checkpoints/onnx/sam2_hiera_small.encoder.onnx \
    --output_decoder checkpoints/onnx/sam2_hiera_small.decoder.onnx \
    --model_type sam2_hiera_small

echo "Converting SAM2-Hiera-BasePlus models..."
python scripts/export_sam2.py --checkpoint checkpoints/sam2_hiera_base_plus.pt \
    --output_encoder checkpoints/onnx/sam2_hiera_base_plus.encoder.onnx \
    --output_decoder checkpoints/onnx/sam2_hiera_base_plus.decoder.onnx \
    --model_type sam2_hiera_base_plus

echo "Converting SAM2-Hiera-Large models..."
python scripts/export_sam2.py --checkpoint checkpoints/sam2_hiera_large.pt \
    --output_encoder checkpoints/onnx/sam2_hiera_large.encoder.onnx \
    --output_decoder checkpoints/onnx/sam2_hiera_large.decoder.onnx \
    --model_type sam2_hiera_large
