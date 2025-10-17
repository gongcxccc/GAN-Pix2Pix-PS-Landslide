import torch

def create_model(opt):
	if opt.model == 'Pix2Pix':
		from .Pix2Pix_model import Pix2PixModel, InferenceModel
		if opt.isTrain:
			model = Pix2PixModel()
		else:
			model = InferenceModel()
	else:
		from .ui_model import UIModel
		model = UIModel()
	model.initialize(opt)
	if opt.verbose:
		print("model [%s] was created" % (model.name()))

	if opt.isTrain and len(opt.gpu_ids) and not opt.fp16:
		model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

	return model
