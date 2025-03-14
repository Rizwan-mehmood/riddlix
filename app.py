import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

# Caching the model loading for faster re-runs
@st.cache_resource
def load_generator():
    model = GPT2LMHeadModel.from_pretrained("./math_riddle_model")
    tokenizer = GPT2Tokenizer.from_pretrained("./math_riddle_model")
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return generator

generator = load_generator()

# Creative Title and Description
st.title("✨ Math Riddle Maestro ✨")
st.write(
    """
    Welcome, curious mind! Prepare to be dazzled as our enchanted model crafts
    five mind-bending math riddles from the depths of numerical mysteries.
    Once generated, we'll spotlight the top three riddles—each a gem in its own right!
    """
)

if st.button("Unleash the Riddles"):
    with st.spinner("Conjuring riddles from the realm of numbers..."):
        # Generate 5 riddles with your fine-tuned model
        results = generator(
            "Riddle:",
            max_length=100,
            num_return_sequences=5,
            temperature=0.5,
            top_k=50
        )

    # Process the results into a list of (riddle, answer) tuples
    riddles = []
    for res in results:
        text = res["generated_text"]
        parts = text.split("Answer:")
        riddle_text = parts[0].strip()
        answer_text = parts[1].strip() if len(parts) > 1 else "???"
        riddles.append((riddle_text, answer_text))

    # Display all generated riddles with creative formatting
    st.subheader("🎲 All Generated Riddles")
    for i, (riddle, answer) in enumerate(riddles, 1):
        st.markdown(f"**Riddle {i}:** {riddle}")
        st.markdown(f"*Proposed Answer:* {answer}")

    # Highlight top 3 riddles (here, we simply choose the first three as an example)
    st.subheader("🌟 Top 3 Riddles")
    for i, (riddle, answer) in enumerate(riddles[:3], 1):
        st.markdown(f"### Riddle {i}")
        st.markdown(f"**{riddle}**")
        st.markdown(f"*Answer:* {answer}")

    st.success("Riddle generation complete! Enjoy the challenge!")
