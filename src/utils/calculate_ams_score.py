import math

def calculate_ams_score(y_true, y_pred, weights, data_split_ratio):
    """Calculates the resulting Approximate Median Significance (AMS) score using the standard formula"""
    weight_multiplier = 1.0 / data_split_ratio 
    
    s = 0.0 
    b = 0.0 
    
    for i in range(len(y_pred)):
        if y_pred[i] == 1: 
            if y_true[i] == 1:
                s += weights[i] * weight_multiplier
            elif y_true[i] == 0:
                b += weights[i] * weight_multiplier
                
    br = 10.0
    if s == 0 and b == 0:
        return 0.0
        
    radicand = 2 * ((s + b + br) * math.log(1.0 + s / (b + br)) - s)
    return math.sqrt(radicand) if radicand > 0 else 0.0
