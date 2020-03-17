import models
import torchvision
import pretrainedmodels

MODEL_DISPATCHER = {
	'se_resnext50': models.SEResNextModel(model=pretrainedmodels.se_resnext50_32x4d()),
	'se_resnext101': models.SEResNextModel(model=pretrainedmodels.se_resnext101_32x4d()),
	'densenet121': models.DensenetModel(model=torchvision.models.densenet121(pretrained=True)),
	'densenet169': models.DensenetModel(model=torchvision.models.densenet169(pretrained=True)),
	'densenet161': models.DensenetModel(model=torchvision.models.densenet161(pretrained=True)),
	'efficientnet-b2': models.EfficientNetModel(model_name='efficientnet-b2'),
	'efficientnet-b3': models.EfficientNetModel(model_name='efficientnet-b3'),
	'efficientnet-b4': models.EfficientNetModel(model_name='efficientnet-b4'),
	'efficientnet-b5': models.EfficientNetModel(model_name='efficientnet-b5'),
}