import joblib


def run_ensemble(ecg_risk,bp_chol_prediction,bp_chol_probability):
    res = []
    # Riskprocent från EKG-modellen
    #ecg_risk = 90  # Risk från EKG-modellen (procent)

    # Input från BP/Chol-modellen
    #bp_chol_prediction = 0  # Prediktion från BP/Chol-modellen (0 eller 1)
    #bp_chol_probability = 0.85  # Sannolikhet från BP/Chol-modellen (0 till 1)

    #ECG_pred = joblib.load('ECG_pred.pkl')
    #BPCh_pred = joblib.load('BPCh_pred.pkl')
    #BPCh_pred_prob = joblib.load('BPCh_pred_prob.pkl')

    #print(ECG_pred)
    #print(BPCh_pred)
    #print(BPCh_pred_prob)


    # Beräkna riskprocent baserat på prediktion och sannolikhet
    if bp_chol_prediction == 0:
        # Om modellen förutspår "inte hälsosam" (0), skala sannolikheten till hög risk
        bp_chol_risk = int(bp_chol_probability * 100)
    else:
        # Om modellen förutspår "hälsosam" (1), invertera sannolikheten till låg risk
        bp_chol_risk = int((1 - bp_chol_probability) * 100)


    # Dynamiska vikter baserat på båda riskerna
    #def adjust_weights(ecg_risk, bp_chol_risk):
        
        #Justera vikterna för de två modellerna baserat på båda risknivåerna.
        #Om en risk är 0, ge hela vikten till den andra.
        
    #    if ecg_risk == 0 and bp_chol_risk == 0:
    #        return 0.5, 0.5  # Ge lika vikt om båda är noll
    #    elif ecg_risk == 0:
    #        return 0.0, 1.0  # All vikt till BP/Chol om EKG-risk är 0
    #    elif bp_chol_risk == 0:
    #        return 1.0, 0.0  # All vikt till EKG om BP/Chol-risk är 0
    #    else:
    #        total_risk = ecg_risk + bp_chol_risk
    #        ecg_weight = ecg_risk / total_risk
    #        bp_chol_weight = bp_chol_risk / total_risk
    #        return ecg_weight, bp_chol_weight

    # Hämta dynamiska vikter
    #ecg_weight, bp_chol_weight = adjust_weights(ecg_risk, bp_chol_risk)

    # Kombinera riskerna med justerade vikter
    #combined_risk = (ecg_risk * ecg_weight + bp_chol_risk * bp_chol_weight)

    

    combined_risk = (1 - (1 - ecg_risk/100) * (1 - bp_chol_risk/100))*100



    # Tröskelvärden för riskkategori
    if combined_risk < 30:
        risk_category = "Low risk for heart disease."
        image = "Green.jpeg"
    elif 30 <= combined_risk < 60:
        risk_category = "Moderate risk for heart disease"
        image = "Yellow.jpeg"
    elif 60 <= combined_risk < 80:
        risk_category = "High risk for heart disease."
        image = "Orange.jpeg"
    else:
        risk_category = "Very high risk for heart disease"
        image = "Red.jpeg"

    # Skriv ut sammanvägt resultat på samma rad
    res = [(f"Combined risk: {combined_risk:.2f}%")]

    # Extra medicinsk insikt om arytmi vid hög EKG-risk
    if ecg_risk >= 70:
        res += ["You have a high degree of arrhythmia, consult a doctor!"]

    # Skriv ut procentuellt resultat för BP/Chol
    #print(f"BP/Chol Risk: {bp_chol_risk}%")
    return res, image

    #return combined_risk, risk_category, ecg_risk



#print(run_ensemble(0.00,1,1))