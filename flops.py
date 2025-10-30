import torch
from thop import profile, clever_format
from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DIDF_Encoder = Restormer_Encoder().to(device)
DIDF_Decoder = Restormer_Decoder().to(device)
BaseFuseLayer = BaseFeatureExtraction(dim=64, num_heads=8).to(device)
DetailFuseLayer = DetailFeatureExtraction(num_layers=1).to(device)

input_size = (1, 1, 256, 256)

data_VIS = torch.randn(input_size).to(device)
data_IR = torch.randn(input_size).to(device)

flops_encoder, params_encoder = profile(DIDF_Encoder, inputs=(data_VIS,), verbose=False)

feature_V_B, feature_V_D, _ = DIDF_Encoder(data_VIS)
flops_decoder, params_decoder = profile(DIDF_Decoder, inputs=(data_VIS, feature_V_B, feature_V_D), verbose=False)

feature_I_B, _, _ = DIDF_Encoder(data_IR)
feature_F_B = BaseFuseLayer(feature_V_B + feature_I_B)
flops_base, params_base = profile(BaseFuseLayer, inputs=(feature_F_B,), verbose=False)

feature_I_D, _, _ = DIDF_Encoder(data_IR)
feature_F_D = DetailFuseLayer(feature_V_D + feature_I_D)
flops_detail, params_detail = profile(DetailFuseLayer, inputs=(feature_F_D,), verbose=False)

total_params = params_encoder + params_decoder + params_base + params_detail
total_flops = flops_encoder + flops_decoder + flops_base + flops_detail

flops_encoder, params_encoder = clever_format([flops_encoder, params_encoder], "%.3f")
flops_decoder, params_decoder = clever_format([flops_decoder, params_decoder], "%.3f")
flops_base, params_base = clever_format([flops_base, params_base], "%.3f")
flops_detail, params_detail = clever_format([flops_detail, params_detail], "%.3f")
total_flops, total_params = clever_format([total_flops, total_params], "%.3f")

print(f"DIDF_Encoder - FLOPs: {flops_encoder}, 参数数量: {params_encoder}")
print(f"DIDF_Decoder - FLOPs: {flops_decoder}, 参数数量: {params_decoder}")
print(f"BaseFuseLayer - FLOPs: {flops_base}, 参数数量: {params_base}")
print(f"DetailFuseLayer - FLOPs: {flops_detail}, 参数数量: {params_detail}")

print(f"\n总 FLOPs: {total_flops}")
print(f"总参数数量: {total_params}")