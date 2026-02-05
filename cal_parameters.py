# 计算参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    from models.model_convnext import fusion_net
    model = fusion_net()
    print(f"模型参数量: {count_parameters(model)}")  # 373192376
