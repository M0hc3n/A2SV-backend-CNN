
def predict(learn, img):
    pred_class, pred_idx, outputs = learn.predict(img)
    
    return pred_class, pred_idx, outputs