from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_PATH = "bert_mitigated_final"

print("Loading DistilBERT model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()

print("Model loaded successfully")


# identity group keywords for fairness correction
identity_words = [
    "muslim","christian","jew","hindu",
    "black","white","asian",
    "gay","lesbian","trans",
    "woman","man","female","male"
]


def analyze_text(text):

    # detect identity mention
    identity_mention = any(word in text.lower() for word in identity_words)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]

    toxicity_score = float(probs[1])

    # fairness correction
    # if identity group mentioned but text not clearly toxic
    if identity_mention and toxicity_score < 0.85:
        toxicity_score = toxicity_score * 0.6

    toxicity = "toxic" if toxicity_score > 0.5 else "safe"

    # placeholder sentiment (can upgrade later)
    sentiment_score = 0.65
    sentiment = "POSITIVE"

    # placeholder bias score
    bias_score = 0.30
    bias = "neutral"

    # overall risk calculation
    risk_score = (toxicity_score + bias_score) / 2

    if risk_score < 0.25:
        risk = "LOW"
    elif risk_score < 0.5:
        risk = "MEDIUM"
    elif risk_score < 0.75:
        risk = "HIGH"
    else:
        risk = "CRITICAL"

    return {
        "toxicity": toxicity,
        "toxicity_score": toxicity_score,
        "sentiment": sentiment,
        "sentiment_score": sentiment_score,
        "bias": bias,
        "bias_score": bias_score,
        "overall_risk": risk,
        "risk_score": risk_score,
        "keywords": [],
        "highlighted_text": text,
        "identity_mention": identity_mention
    }