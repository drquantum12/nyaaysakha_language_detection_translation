# from lang_det.Inference.main import detect_language
from fasttext import load_model 

# Load the pre-trained model
model = load_model('../models/lid.176.bin')

def detect_language(text):
    # Predict the language of the text
    predictions = model.predict(text)
    return predictions

if __name__ == "__main__":
    test_samples = [
   'আহকচোন!',
   'aaj key din ka mausam atyant sundar hai, jahan sadaiv chae hue baadal, gulabi rangeen shaam, aur halki havaa key saath praakritik saundarya kaa anand lene kaa aeka sunhara avsar haye',
    ]
    sample = test_samples[1]
    output = detect_language(sample)
    print(output)