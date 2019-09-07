from dl_cv.neuralnetwork import LeNet
from dl_cv.neuralnetwork import VisualizeArchitecture

model = LeNet.build(28, 28, 1, 10)
VisualizeArchitecture.visualize(model, "lenet_architecture.png")