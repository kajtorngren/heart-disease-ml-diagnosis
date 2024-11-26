import joblib
def run_ensemble(data):
    # Riskprocent från EKG-modellen
    ecg_risk = 90  # Risk från EKG-modellen (procent)

    # Input från BP/Chol-modellen
    bp_chol_prediction = 1  # Prediktion från BP/Chol-modellen (0 eller 1)
    bp_chol_probability = 0.76  # Sannolikhet från BP/Chol-modellen (0 till 1)

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

    # Dynamiska vikter baserat på medicinsk kontext
    def adjust_weights(ecg_risk, bp_chol_risk):
        """
        Justera vikterna för de två modellerna baserat på risknivåer
        """
        # Om BP/Chol-risk är hög, ge BP-modellen mer vikt
        if bp_chol_risk >= 70:
            return 0.4, 0.6  # Ge mer vikt till BP/Chol om BP är mycket hög
        # Om både EKG och BP/Chol är höga, ge lika vikt
        elif ecg_risk >= 70 and bp_chol_risk >= 70:
            return 0.5, 0.5  # Ge lika vikt om både EKG och BP är höga
        # Om både EKG och BP/Chol är låga, ge lika vikt
        elif ecg_risk < 30 and bp_chol_risk < 50:
            return 0.5, 0.5  # Låg risk för båda, ge lika vikt
        else:
            return 0.5, 0.5  # Standard när riskerna är balanserade

    # Hämta dynamiska vikter
    ecg_weight, bp_chol_weight = adjust_weights(ecg_risk, bp_chol_risk)

    # Kombinera riskerna med justerade vikter
    combined_risk = (ecg_risk * ecg_weight + bp_chol_risk * bp_chol_weight)

    # Tröskelvärden för riskkategori
    if combined_risk < 50:
        risk_category = "Låg risk för hjärtsjukdom"
    elif 50 <= combined_risk < 75:
        risk_category = "Medelhög risk för hjärtsjukdom"
    elif 75 <= combined_risk < 90:
        risk_category = "Hög risk för hjärtsjukdom"
    else:
        risk_category = "Mycket hög risk för hjärtsjukdom"

    # Skriv ut sammanvägt resultat på samma rad
    print(f"Kombinerad risk: {combined_risk:.2f}% - {risk_category}")

    # Extra medicinsk insikt om arytmi vid hög EKG-risk
    if ecg_risk >= 70:
        print("Du har stor grad av arytmi, uppsök läkare")

    # Skriv ut procentuellt resultat för BP/Chol
    print(f"BP/Chol Risk: {bp_chol_risk}%")
    return 0


print(run_ensemble(1))