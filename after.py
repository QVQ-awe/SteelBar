import torch
from ultralytics.nn.tasks import DetectionModel
from torch.nn.modules.container import Sequential

# 加入白名单
torch.serialization.add_safe_globals([DetectionModel, Sequential])

# 加载你这次训练的last.pt
model = torch.load(r"F:\programing\SteelBar\runs\detect\train5\weights\last.pt", map_location="cpu", weights_only=False)
# 重新保存为干净的模型
torch.save(model["model"].state_dict(), r"F:\programing\SteelBar\fixed_best.pt")

print("✅ 模型修复完成！可以直接用fixed_best.pt推理了")