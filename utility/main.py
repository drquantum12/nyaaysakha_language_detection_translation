from models.translate import batch_translate

def get_translation(input_text, src_lang, tgt_lang, model, tokenizer, ip):
    translations = batch_translate([input_text], src_lang, tgt_lang, model, tokenizer, ip)
    return translations[0]