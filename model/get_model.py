import segformer

def get_segformer_multiTWA(num_classes=2, phi="b2", pretrained=True):
    
    model = segformer.SegFormer(
        num_classes = num_classes,
        phi = phi,
        pretrained = pretrained,
        )
    
    return model