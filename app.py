import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM, pipeline
from PIL import Image
import torch
import io

# Force CPU mode (Streamlit Cloud usually has no GPU)
device = "cpu"

@st.cache_resource
def load_models():
    # Image captioning
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        dtype="float32"
    ).to(device)

    # Story generation (Gemma fine-tuned model)
    story_model_id = "DesuChan123/Gemma-Generate-Story-Model"
    text_tokenizer = AutoTokenizer.from_pretrained(story_model_id)
    text_model = AutoModelForCausalLM.from_pretrained(
        story_model_id,
        dtype="float32"
    ).to(device)

    # Text-to-speech
    tts = pipeline("text-to-speech", model="facebook/mms-tts-eng")

    return blip_processor, blip_model, text_tokenizer, text_model, tts

blip_processor, blip_model, text_tokenizer, text_model, tts = load_models()

# Functions
def img2caption(image_file):
    raw_image = Image.open(image_file).convert("RGB")
    inputs = blip_processor(raw_image, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs, max_new_tokens=30)
    return blip_processor.decode(out[0], skip_special_tokens=True)

def generate_story(caption):
    prompt = (
        f"Write a bedtime story for children aged 3–10 based on this caption: {caption}. "
        f"Make sure the story has a beginning, middle, and end, is 50–100 words long, "
        f"and finishes naturally."
    )
    inputs = text_tokenizer(prompt, return_tensors="pt").to(device)
    output = text_model.generate(
        **inputs,
        max_new_tokens=220,
        min_length=80,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        pad_token_id=text_tokenizer.eos_token_id,
    )
    story = text_tokenizer.decode(output[0], skip_special_tokens=True).strip()
    return story

def story_to_audio(story_text):
    audio_out = tts(story_text)
    return audio_out["audio"], audio_out["sampling_rate"]

# Streamlit UI
st.title("📖 Kids Storytelling App")
st.write("Upload an image and let AI create a magical story with narration!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Story"):
        caption = img2caption(uploaded_file)
        st.subheader("Generated Caption")
        st.write(caption)

        story = generate_story(caption)
        st.subheader("Generated Story")
        st.write(story)

        audio, rate = story_to_audio(story)
        st.subheader("Listen to the Story")
        st.audio(io.BytesIO(audio), format="audio/wav")
