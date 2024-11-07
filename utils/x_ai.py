from captum.attr import *

def getInputAttributions(model, input_tensor,target):
    ig = IntegratedGradients(model)
    input_tensor.requires_grad_()
    # attr, delta = ig.attribute(input_tensor, target=target, return_convergence_delta=True)
    attr = ig.attribute(input_tensor)
    attr = attr.cpu().detach().numpy()
    return attr

def getInputAttributions_DeepLift(model, input_tensor,target):
    ig = DeepLift(model,multiply_by_inputs=True)
    input_tensor.requires_grad_()
    attr, delta = ig.attribute(input_tensor, target=target, return_convergence_delta=True)
    attr = attr.cpu().detach().numpy()
    return attr
