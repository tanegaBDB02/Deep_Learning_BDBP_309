import torch
import torchvision.models as models

# Create model and load pretrained weights
model = models.vgg16(weights='IMAGENET1K_V1')

# Save only weights
torch.save(model.state_dict(), 'model_weights.pth')

# Later: recreate the model (without pretrained weights)
model = models.vgg16()
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()