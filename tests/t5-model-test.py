from transformers import T5ForConditionalGeneration

MODEL_PATH = "data/models/t5-large"

model: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(
    MODEL_PATH
)
print(model.__class__)

# https://github.com/LengSicong/Tell2Design/blob/main/Imagen/imagen.py#L924-L970
