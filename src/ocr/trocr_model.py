from transformers import TrOCRProcessor, VisionEncoderDecoderModel

def load_model(model_dir="models/trocr_finetuned"):
    processor = TrOCRProcessor.from_pretrained(model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id

    return processor, model

def load_base_model():
    return load_model("microsoft/trocr-base-handwritten")