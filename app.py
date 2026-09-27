# ============================================================
# Children's Story Generator - Image -> Story -> Audio
# Streamlit Cloud Ready Version
# ============================================================
import io
import numpy as np
import scipy.io.wavfile as wavfile
import streamlit as st
from PIL import Image
from transformers import pipeline


@st.cache_resource
def load_captioner():
    """Load BLIP image captioning model from Hugging Face Hub."""
    return pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base"
    )


@st.cache_resource
def load_story_generator():
    """Load GPT-2 text generation model from Hugging Face Hub."""
    return pipeline(
        "text-generation",
        model="gpt2"
    )


@st.cache_resource
def load_tts():
    """Load MMS-TTS English model from Hugging Face Hub."""
    return pipeline(
        "text-to-speech",
        model="facebook/mms-tts-eng"
    )


def generate_caption(captioner, image):
    """Generate a short caption from the uploaded image."""
    result = captioner(image)
    return result[0]["generated_text"]


def generate_story(story_generator, caption):
    """Expand the caption into a short children's story."""
    prompt = (
        f"Write a short, happy story for children (50-100 words). "
        f"The story is about: {caption}. Story:"
    )
    result = story_generator(
        prompt,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1
    )
    story = result[0]["generated_text"]
    story = story.replace(prompt, "").strip()
    return story


def generate_audio(tts, story):
    """Convert the story text into a WAV audio buffer."""
    speech = tts(story)
    audio_array = np.array(speech["audio"]).squeeze()
    sample_rate = speech["sampling_rate"]

    wav_buffer = io.BytesIO()
    wavfile.write(wav_buffer, sample_rate, audio_array)
    wav_buffer.seek(0)
    return wav_buffer


def main():
    """Main Streamlit application entry point."""
    st.set_page_config(
        page_title="Children's Story Generator",
        page_icon="🧸",
        layout="centered"
    )

    st.title("🧸 Children's Story Generator")
    st.caption("Upload an image, and AI will tell you a fun little story!")

    with st.sidebar:
        st.header("ℹ️ About This App")
        st.markdown("""
        - **Image Captioning**: `Salesforce/blip-image-captioning-base`
        - **Story Generation**: `gpt2`
        - **Text-to-Speech**: `facebook/mms-tts-eng`
        """)
        st.divider()
        st.caption("🎈 Suitable for children aged 3-10")

    with st.spinner("Loading models, this may take a few minutes on first run..."):
        try:
            captioner = load_captioner()
            story_generator = load_story_generator()
            tts = load_tts()
            st.success("✅ Models loaded successfully")
        except Exception as e:
            st.error(f"❌ Model loading failed: {str(e)}")
            st.stop()

    uploaded_file = st.file_uploader(
        "📤 Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Your uploaded image", use_container_width=True)

        if st.button("✨ Generate Story", type="primary"):
            try:
                with st.spinner("Looking at the image..."):
                    caption = generate_caption(captioner, image)
                    st.info(f"📷 Image caption: {caption}")

                with st.spinner("Writing the story..."):
                    story = generate_story(story_generator, caption)

                st.subheader("📖 Generated Story")
                st.write(story)

                with st.spinner("Generating audio..."):
                    wav_buffer = generate_audio(tts, story)

                st.subheader("🔊 Audio Version")
                st.audio(wav_buffer, format="audio/wav")

            except Exception as e:
                st.error(f"An error occurred during generation: {str(e)}")

    st.divider()
    with st.expander("🔧 Technical Details"):
        st.markdown("""
        **Pipeline Flow**: `Image Upload` → `BLIP Captioning` → `GPT-2 Story Generation` → `Hugging Face TTS`

        **Three Pipelines**:
        1. `image-to-text`: Salesforce/blip-image-captioning-base
        2. `text-generation`: gpt2
        3. `text-to-speech`: facebook/mms-tts-eng
        """)


if __name__ == "__main__":
    main()
