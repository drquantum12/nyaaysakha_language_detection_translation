from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from pydantic import BaseModel
from models.translate import initialize_model_and_tokenizer, batch_translate
import torch
from models.IndicTransToolkit.IndicTransToolkit.processor import IndicProcessor
from lang_det.Inference.ai4bharat.IndicLID import IndicLID
import numpy as np
ml_models = {}

@asynccontextmanager
async def lifespan(app:FastAPI):
    print("Start of lifespan")
    BATCH_SIZE = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    quantization = None
    print("Using device:", DEVICE)

    # for language detection
    ml_models['IndicLID_model'] = IndicLID(input_threshold = 0.5, roman_lid_threshold = 0.6)

    # for translation
    en_indic_ckpt_dir = "ai4bharat/indictrans2-en-indic-dist-200M"  # ai4bharat/indictrans2-en-indic-dist-200M ai4bharat/indictrans2-en-indic-1B
    indic_en_ckpt_dir = "ai4bharat/indictrans2-indic-en-dist-200M"
    ml_models['en_indic_tokenizer'], ml_models['en_indic_model'] = initialize_model_and_tokenizer(en_indic_ckpt_dir, quantization, DEVICE=DEVICE)
    ml_models['indic_en_tokenizer'], ml_models['indic_en_model'] = initialize_model_and_tokenizer(indic_en_ckpt_dir, quantization, DEVICE=DEVICE)
    ml_models['ip'] = IndicProcessor(inference=True)

    print("Models loaded successfully!")

    yield
    ml_models.clear()
    print("End of lifespan")

app = FastAPI(lifespan=lifespan)

# getting user text input
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Machine Translation API!"}

class UserInput(BaseModel):
    input_text: str

# getting user input as text from body of the request
@app.post("/translate/")
async def translate_text(user_input: UserInput):
    resp = {}
    translation_required = None
    try:
        target_lang = "eng_Latn"
        text = user_input.input_text
        language_detected = ml_models["IndicLID_model"].batch_predict([text], 1)[0][-3]
        print(language_detected)
        if not language_detected.startswith("en"):
            translation_required = True
            target_lang = "eng_Latn"
            model = ml_models["indic_en_model"]
            tokenizer = ml_models["indic_en_tokenizer"]
            src_lang = language_detected
        print(f"Translation required: {translation_required}")
        if translation_required:
            translations = batch_translate([text], src_lang, target_lang, model, tokenizer, ml_models["ip"], BATCH_SIZE=1, DEVICE="cpu")
            print(translations)
            resp["Translation"] = translations[0]
        else:
            resp["Translation"] = text
        resp["Detected Language"] = language_detected
        resp["Translation Required"] = translation_required
        resp["Target Language"] = target_lang
        print(resp)
        return resp
    except Exception as e:
        return {"error": str(e)}