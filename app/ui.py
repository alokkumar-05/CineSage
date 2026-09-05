
# pyrefly: ignore [missing-import]
import streamlit as st
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from langchain_core.prompts import ChatPromptTemplate
# pyrefly: ignore [missing-import]
from pydantic import BaseModel
from typing import List, Optional
# pyrefly: ignore [missing-import]
from langchain_core.output_parsers import PydanticOutputParser
# pyrefly: ignore [missing-import]
from langchain_mistralai import ChatMistralAI

# -------------------- Setup & Page Config --------------------
st.set_page_config(page_title="🎬 CineSage - Movie Info Extractor", page_icon="🎬", layout="centered")

load_dotenv()

@st.cache_resource
def get_model():
    return ChatMistralAI(model="open-mistral-nemo")

model = get_model()

# -------------------- Schema --------------------
class Movie(BaseModel):
    title: str
    release_year: Optional[int] = None
    genre: List[str] = []
    director: Optional[str] = None
    cast: List[str] = []
    rating: Optional[float] = None
    summary: str

parser = PydanticOutputParser(pydantic_object=Movie)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
Extract movie information from the paragraph.
{format_instructions}
"""),
    ("human", "{paragraph}")
])

# -------------------- UI --------------------
st.title("🎬 CineSage: Movie Extractor")
st.caption("Paste any movie description and AI will extract structured information.")

sample_text = (
    "The Martian is a 2015 science-fiction film directed by Ridley Scott and starring Matt Damon, "
    "Jessica Chastain, Kristen Wiig, Jeff Daniels, Michael Peña, and Sean Bean. The story follows "
    "Mark Watney, an astronaut who is mistakenly presumed dead and left behind on Mars. With limited "
    "supplies, Watney must use his ingenuity to survive by growing food and producing essential "
    "resources. Meanwhile, NASA and his crew work to find a way to rescue him."
)

col_sample, col_clear = st.columns([1, 4])
if col_sample.button("Load Sample Text"):
    st.session_state["movie_input"] = sample_text

paragraph = st.text_area(
    "Enter Movie Paragraph",
    value=st.session_state.get("movie_input", ""),
    height=180,
    placeholder="Paste a movie synopsis, review, or article here..."
)

if st.button("🚀 Extract Data", type="primary", use_container_width=True):
    if not paragraph.strip():
        st.warning("Please enter a paragraph first.")
    else:
        with st.spinner("Analyzing movie details with AI..."):
            try:
                final_prompt = prompt.invoke({
                    "paragraph": paragraph,
                    "format_instructions": parser.get_format_instructions()
                })

                response = model.invoke(final_prompt)
                movie_data = parser.parse(response.content)
                movie_dict = movie_data.model_dump() if hasattr(movie_data, "model_dump") else movie_data.dict()

                st.success("Extraction Completed Successfully!")

                # Structured Details Display
                st.subheader(f"🍿 {movie_data.title} ({movie_data.release_year or 'N/A'})")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**🎬 Director:** {movie_data.director or 'N/A'}")
                    st.markdown(f"**⭐ Rating:** {movie_data.rating if movie_data.rating is not None else 'N/A'}")
                    st.markdown(f"**🎭 Genres:** {', '.join(movie_data.genre) if movie_data.genre else 'N/A'}")
                with col2:
                    st.markdown(f"**👥 Cast:** {', '.join(movie_data.cast) if movie_data.cast else 'N/A'}")
                
                st.markdown(f"**📝 Summary:** {movie_data.summary}")

                st.divider()

                with st.expander("🔍 View JSON & Raw Model Response"):
                    st.subheader("Structured JSON")
                    st.json(movie_dict)
                    st.subheader("Raw Model Output")
                    st.code(response.content, language="json")

            except Exception as e:
                st.error("Failed to parse response. Model did not follow schema.")
                st.exception(e)