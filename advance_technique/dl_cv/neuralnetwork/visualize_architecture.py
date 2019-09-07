from keras.utils import plot_model

class VisualizeArchitecture:
    @staticmethod
    def visualize(model, figName):
        plot_model(model, to_file=figName, show_shapes=True)