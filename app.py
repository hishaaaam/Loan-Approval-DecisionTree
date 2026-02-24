import gradio as gr
import joblib
import numpy as np
import os

# =============================
# LOAD MODEL
# =============================
MODEL_PATH = "model/model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "model.pkl not found. Upload your trained model inside /model folder."
    )

model = joblib.load(MODEL_PATH)


# =============================
# PREDICTION FUNCTION
# =============================
def predict(income, credit_score, loan_amount, age, employment_years):
    try:
        features = np.array([[
            income,
            credit_score,
            loan_amount,
            age,
            employment_years
        ]])

        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0][pred]
        confidence = round(float(prob) * 100, 2)

        decision = "✅ Approved" if pred == 1 else "❌ Rejected"
        badge = "🟢" if pred == 1 else "🔴"

        message = (
            f"## {badge} Loan Decision\n\n"
            f"### **{decision}**\n\n"
            f"**Confidence:** {confidence}%"
        )

        return message, confidence

    except Exception as e:
        return f"❌ Error: {str(e)}", 0


# =============================
# CUSTOM CSS
# =============================
CUSTOM_CSS = """
.gradio-container {
    font-family: 'Inter', sans-serif;
}

.main-header {
    text-align: center;
    padding: 30px 10px;
    background: linear-gradient(135deg, #4f46e5, #06b6d4);
    border-radius: 16px;
    color: white;
    margin-bottom: 20px;
}

.card {
    background: #ffffff;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
}

.footer {
    text-align: center;
    color: #888;
    margin-top: 20px;
    font-size: 14px;
}
"""


# =============================
# UI
# =============================
with gr.Blocks(css=CUSTOM_CSS, title="AI Loan Approval System") as demo:

    gr.HTML(
        """
        <div class="main-header">
            <h1>💳 AI Loan Approval System</h1>
            <p>Decision Tree–powered credit risk assessment</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column():
            with gr.Group(elem_classes="card"):
                gr.Markdown("### 🧾 Applicant Details")

                income = gr.Number(label="Annual Income", value=60000)
                credit_score = gr.Number(label="Credit Score (300–850)", value=700)
                loan_amount = gr.Number(label="Loan Amount", value=20000)
                age = gr.Number(label="Age", value=30)
                employment_years = gr.Number(label="Employment Years", value=4)

                predict_btn = gr.Button("🔍 Evaluate Application", size="lg")

        with gr.Column():
            with gr.Group(elem_classes="card"):
                gr.Markdown("### 📊 Decision Output")

                output_text = gr.Markdown(
                    "Submit applicant details to see decision."
                )

                confidence_bar = gr.Slider(
                    label="Confidence Score (%)",
                    minimum=0,
                    maximum=100,
                    value=0,
                )

    gr.HTML(
        '<div class="footer">Built with Decision Tree • Gradio • Hugging Face Spaces</div>'
    )

    predict_btn.click(
        fn=predict,
        inputs=[income, credit_score, loan_amount, age, employment_years],
        outputs=[output_text, confidence_bar],
    )


# =============================
# LAUNCH (Spaces safe)
# =============================
if __name__ == "__main__":
    demo.launch()