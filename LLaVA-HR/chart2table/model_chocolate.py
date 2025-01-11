from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image

model_name = "khhuang/chart-to-table"
model = VisionEncoderDecoderModel.from_pretrained(model_name).cuda()
processor = DonutProcessor.from_pretrained(model_name)

image_path = "images/ai4sci/co2.png"

# Format text inputs

input_prompt = "<data_table_generation> <s_answer>"

# Encode chart figure and tokenize text
img = Image.open(image_path)
pixel_values = processor(img.convert("RGB"), random_padding=False, return_tensors="pt").pixel_values
pixel_values = pixel_values.cuda()
decoder_input_ids = processor.tokenizer(
    input_prompt, add_special_tokens=False, return_tensors="pt", max_length=510
).input_ids.cuda()  # .squeeze(0)


outputs = model.generate(
    pixel_values.cuda(),
    decoder_input_ids=decoder_input_ids.cuda(),
    max_length=model.decoder.config.max_position_embeddings,
    early_stopping=True,
    pad_token_id=processor.tokenizer.pad_token_id,
    eos_token_id=processor.tokenizer.eos_token_id,
    use_cache=True,
    num_beams=4,
    bad_words_ids=[[processor.tokenizer.unk_token_id]],
    return_dict_in_generate=True,
)


sequence = processor.batch_decode(outputs.sequences)[0]
sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
extracted_table = sequence.split("<s_answer>")[1].strip()
extracted_table = extracted_table.replace("&&&", "|\n|")

print(extracted_table)
