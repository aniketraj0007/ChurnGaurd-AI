def get_risk_level(probability):
    """Classifies churn probability into risk categories."""
    if probability >= 0.60:
        return "High Risk"
    elif probability >= 0.30:
        return "Medium Risk"
    else:
        return "Low Risk"


def get_retention_action(customer_dict):
    """
    Generates practical retention recommendations based on simple customer attributes.
    """
    contract = customer_dict.get("Contract", "Month-to-month")
    monthly_charges = float(customer_dict.get("MonthlyCharges", 0))
    tenure = int(customer_dict.get("tenure", 0))
    tech_support = customer_dict.get("TechSupport", "No")
    payment_method = customer_dict.get("PaymentMethod", "Electronic check")
    
    actions = []
    
    if contract == "Month-to-month":
        actions.append("Offer a contract upgrade discount (1-Year or 2-Year plan)")
        
    if monthly_charges > 70.0:
        actions.append("Provide a personalized monthly bill discount or loyalty credit")
        
    if tech_support in ["No", "No internet service"]:
        actions.append("Offer a complimentary technical support package")
        
    if tenure < 12:
        actions.append("Schedule a new customer onboarding / satisfaction check-in call")
        
    if payment_method == "Electronic check":
        actions.append("Encourage setup of automatic bank/card payment with a one-time bill credit")
        
    if not actions:
        actions.append("Send standard customer appreciation & loyalty rewards offer")
        
    return actions
