import gradio as gr
import joblib
import pandas as pd

# Load trained model
model = joblib.load("customer_churn_model.pkl")

# Prediction function
def predict_churn(
    gender, age, partner, dependents, tenure,
    monthly_bill, total_bill,
    fiber_optic, internet_no,
    contract_2years, contract_monthly,
    payment_credit_card, payment_electronic_check,
    payment_mailed_check
):
    
    input_data = pd.DataFrame([[
        gender, age, partner, dependents, tenure,
        monthly_bill, total_bill,
        fiber_optic, internet_no,
        contract_2years, contract_monthly,
        payment_credit_card, payment_electronic_check,
        payment_mailed_check
    ]], columns=[
        "Gender",
        "Age",
        "Partner",
        "Dependents",
        "Tenure",
        "MonthlyBill",
        "TotalBill",
        "InternetService_Fiber optic",
        "InternetService_No",
        "ContractType_2 Years",
        "ContractType_Monthly",
        "PaymentMethod_Credit card",
        "PaymentMethod_Electronic check",
        "PaymentMethod_Mailed check"
    ])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        return "Churn: Yes"
    else:
        return "Churn: No"


# Gradio Frontend
demo = gr.Interface(
    fn=predict_churn,
    inputs=[
        gr.Number(label="Gender"),
        gr.Number(label="Age"),
        gr.Number(label="Partner"),
        gr.Number(label="Dependents"),
        gr.Number(label="Tenure"),
        gr.Number(label="Monthly Bill"),
        gr.Number(label="Total Bill"),
        gr.Number(label="Internet Service - Fiber optic"),
        gr.Number(label="Internet Service - No"),
        gr.Number(label="Contract Type - 2Years"),
        gr.Number(label="Contract Type - Monthly"),
        gr.Number(label="Payment Method - Credit card"),
        gr.Number(label="Payment Method - Electronic check"),
        gr.Number(label="Payment Method - Mailed check")
    ],
    outputs=gr.Textbox(label="Churn Prediction"),
    title="Customer Churn Prediction"
)

# Launch application
demo.launch()