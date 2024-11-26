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
    def adjust_weights(ecg_risk, bp_chol_risk):
        """
        Justera vikterna för de två modellerna baserat på båda risknivåerna.
        Om en risk är 0, ge hela vikten till den andra.
        """
        if ecg_risk == 0 and bp_chol_risk == 0:
            return 0.5, 0.5  # Ge lika vikt om båda är noll
        elif ecg_risk == 0:
            return 0.0, 1.0  # All vikt till BP/Chol om EKG-risk är 0
        elif bp_chol_risk == 0:
            return 1.0, 0.0  # All vikt till EKG om BP/Chol-risk är 0
        else:
            total_risk = ecg_risk + bp_chol_risk
            ecg_weight = ecg_risk / total_risk
            bp_chol_weight = bp_chol_risk / total_risk
            return ecg_weight, bp_chol_weight

    # Hämta dynamiska vikter
    ecg_weight, bp_chol_weight = adjust_weights(ecg_risk, bp_chol_risk)

    # Kombinera riskerna med justerade vikter
    combined_risk = (ecg_risk * ecg_weight + bp_chol_risk * bp_chol_weight)

    # Tröskelvärden för riskkategori
    if combined_risk < 50:
        risk_category = "Låg risk för hjärtsjukdom."
    elif 50 <= combined_risk < 75:
        risk_category = "Medelhög risk för hjärtsjukdom."
    elif 75 <= combined_risk < 90:
        risk_category = "Hög risk för hjärtsjukdom."
    else:
        risk_category = "Mycket hög risk för hjärtsjukdom."

    # Skriv ut sammanvägt resultat på samma rad
    res = [(f"Kombinerad risk: {combined_risk:.2f}% - {risk_category}")]

    # Extra medicinsk insikt om arytmi vid hög EKG-risk
    if ecg_risk >= 70:
        res += ["Du har stor grad av arytmi, uppsök läkare!"]

    # Skriv ut procentuellt resultat för BP/Chol
    #print(f"BP/Chol Risk: {bp_chol_risk}%")
    return res




#print(run_ensemble(0.00,1,1))