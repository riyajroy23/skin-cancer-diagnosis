import tensorflowjs as tfjs

def save_tfjs(model, path):
    tfjs.converters.save_keras_model(model, path)