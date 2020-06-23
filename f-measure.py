def ubahlabel(labeluji):
    y = []
    for i in range(len(labeluji)):
        if labeluji[i] == 'pribadi':
            y.append(0)
        elif labeluji[i] == 'promosi':
            y.append(1)
        else:
            y.append(2)
    return y
            
y_pred = ubahlabel(hasiluji)
y_true = ubahlabel(labelujisebenarnya)
print(y_pred)
print(labelujisebenarnya)
print(y_true)

#MICROAVERAGE
def microaverage(y_true, y_pred):
    TP = 0
    TPFP = 0
    for i in range(len(y_true)):
        TPFP +=1
        if y_true[i] == y_pred[i]:
            TP+=1
    return(TP/TPFP)
print('Microaverage: {0} \n'.format(microaverage(y_true, y_pred)))

#MACROAVERAGE
def macroaverage(y_true, y_pred,PoR):
    TP = []
    FP = []
    if PoR == 'precision':
        a = y_pred
        b = y_true
    elif PoR == 'recall':
        a = y_true
        b = y_pred
        
    for i in range(3):
        TP_S = 0
        FP_S = 0
        for j in range(len(y_true)):
            if a[j] == i:
                if b[j] == i:
                    TP_S += 1
                else:
                    FP_S += 1
        TP.append(TP_S)
        FP.append(FP_S)
        
    macro = 0
    m = 0
    for i in range(3):
        m = (TP[i]/(TP[i]+FP[i]))
        macro += m
    return(macro/3)

macrorecall = macroaverage(y_true, y_pred,'recall')
macroprecision = macroaverage(y_true, y_pred,'precision')
macrof1 = 2*(macrorecall*macroprecision)/(macrorecall+macroprecision)

print('Macroaverage Recall: {0}'.format(macrorecall))
print('Macroaverage Precision: {0}'.format(macroprecision))
print('Macro F1-Score: {0}'.format(macrof1))
