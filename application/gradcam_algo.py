import torch
import torch.nn.functional as F

class GradCAM:
    """
    Class to generate GradCAM images. This class takes a model and a layer name as input and generates GradCAM images. This class also has a __call__ method which allows the class to be called like a function. 
    """
    def __init__(self,model,layer_name):

        self.model=model
        self.forward_features=None
        self.backward_gradients=None
        self.cam_layer=getattr(self.model,layer_name)
        self.cam_layer[-1].register_forward_hook(self.forward_features_hook)
        self.cam_layer[-1].register_full_backward_hook(self.backward_gradients_hook)
    
    def forward_features_hook(self,module,input,output):

        self.forward_features=output

    def backward_gradients_hook(self,module,grad_input,grad_output):

        self.backward_gradients=grad_output[0]
    
    def __call__(self, x):

        return self.model(x)
    
    def generata_grad_cam(self):
        layer_weights=F.adaptive_avg_pool2d(self.backward_gradients,(1,1))
        grad_cam=torch.mul(self.forward_features,layer_weights).sum(dim=1,keepdim=True)
        grad_cam=F.relu(grad_cam)
        grad_cam_image=F.interpolate(grad_cam, size=(224,224), mode='bilinear', align_corners=False)
        grad_cam_image=grad_cam_image-torch.min(grad_cam_image)
        grad_cam_image=grad_cam_image/torch.max(grad_cam_image)
        return grad_cam_image