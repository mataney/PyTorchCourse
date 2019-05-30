# pylint: disable=invalid-name

import torch
import torchvision

# Use a pre defined model without control-flows
model = torchvision.models.resnet18()

# In order to save the model it requires an example of possible imports
example = torch.rand(1, 3, 224, 224)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)
# Save
traced_script_module.save("../model.pt")

# Just for fun, let's see what we get if we feed our model a image of 1s.
# output is the unnormalized probability distribution over 1000 resnet classes.
output = traced_script_module(torch.ones(1, 3, 224, 224))

print(output[0, :5])
