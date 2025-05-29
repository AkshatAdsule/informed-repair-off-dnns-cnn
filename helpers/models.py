import sytorch as st
import torchvision


def _deflat_sequeezenet(net):
    layers = []
    for layer in net[0]:
        if isinstance(layer, st.nn.Sequential):
            for m in layer:
                layers.append(m)
        else:
            layers.append(layer)

    for layer in net[1]:
        layers.append(layer)
    layers.append(net[2])
    return st.nn.Sequential(*layers)


def squeezenet(pretrained=True, eval=True, flatten=False):
    network = torchvision.models.squeezenet1_1(pretrained=pretrained).train(
        mode=not eval
    )
    network = st.nn.from_torch(network)
    # network.propagate_prefix()
    if flatten:
        network = _deflat_sequeezenet(network)
    return network
