import os
import streamlit as st
from openai import OpenAI
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# === Page setup ===
st.set_page_config(page_title="Training Research Q&A", layout="centered")
st.title("Strength, Power & Hypertrophy Research Assistant")
st.markdown("Ask questions based on publicly available training research papers.")

# === Load API key from secrets.toml ===
api_key = os.getenv("GROQ_API_KEY") or st.secrets["GROQ_API_KEY"]

# === Set up Groq LLaMA 3 client ===
client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

# === Load vector store once ===
@st.cache_resource
def load_retriever():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local("vector_store", embedding_model, allow_dangerous_deserialization=True)
    return vector_store.as_retriever()

retriever = load_retriever()

# === Answer generation ===
def generate_answer(query):
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs[:3]])

    messages = [
        {"role": "system", "content": "You are a helpful assistant that summarizes sports science training research papers."},
        {"role": "user", "content": f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}"}
    ]

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=messages,
        temperature=0.3,
        max_tokens=512
    )

    return response.choices[0].message.content.strip(), docs

# === User input ===
query = st.text_input("💬 Ask a training question:")
if query:
    with st.spinner("Thinking..."):
        answer, docs = generate_answer(query)

    st.markdown("### ✅ Answer")
    st.write(answer)

    st.markdown("### 📚 References")
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown source")
        st.markdown(f"- Document {i+1}: `{source}`")
# === Footer Disclaimer ===
st.markdown("""<hr style="margin-top: 2em; margin-bottom: 1em;">""", unsafe_allow_html=True)
with st.expander("📘 Disclaimer: About the Data Used"):
    st.markdown("""
    This application uses Retrieval-Augmented Generation (RAG) to answer questions based on **publicly available research papers** in the field of strength, power, and hypertrophy training.  
    The app currently includes insights from the following works:


### References

1. **Schoenfeld, B.J.** (2010). *The mechanisms of muscle hypertrophy and their application to resistance training*. *J Strength Cond Res*.

2. **Schoenfeld, B.J., Ogborn, D., Krieger, J.W.** (2016). *Resistance Training Volume Enhances Muscle Hypertrophy But Not Strength in Trained Men*. *Med Sci Sports Exerc*.

3. **Grgic, J., Schoenfeld, B.J., Latella, C.** (2019). *Resistance training frequency and skeletal muscle hypertrophy: A review*. *Sports Med*.

4. **Krzysztofik, M., Wilk, M., Wojdala, G., Gołaś, A.** (2019). *Maximizing muscle hypertrophy: A systematic review of advanced resistance training techniques and methods*. *J Hum Kinet*.

5. **Suchomel, T.J., Nimphius, S., Stone, M.H.** (2016). *The importance of muscular strength: Training considerations*. *Sports Med*.

6. **Weakley, J.J.S., Mann, B., Banyard, H.G., et al.** (2021). *Velocity-based training: From theory to application*. *Strength Cond J*.

7. **McBride, J.M., Cormie, P., Deane, R.** (2007). *Power vs strength-power jump squat training: Influence on the load-power relationship*. *J Strength Cond Res*.

8. **Rumpf, M.C., Lockie, R.G., Cronin, J.B., Jalilvand, F.** (2016). *Effect of different sprint training methods: A brief review*. *J Strength Cond Res*.

9. **Petrakos, G., Morin, J.B., Egan, B.** (2016). *Resisted sled sprint training: A systematic review*. *Sports Med*.

10. **Barr, M.J., Sheppard, J.M., Agar-Newman, D.J., Newton, R.U.** (2014). *Transfer effect of strength and power training to the sprinting kinematics of international rugby players*. *J Strength Cond Res*.

11. **Bazyler, C.D., Abbott, H., Bellon, C.R., et al.** (2017). *Strength training for endurance athletes: Theory to practice*. *Strength Cond J*.

12. **Loturco, I., Kobal, R., Kitamura, K., et al.** (2015). *Predicting the maximum dynamic strength in bench press: The high precision of the bar velocity approach*. *J Strength Cond Res*.

13. **Wilson, J.M., Marin, P.J., Rhea, M.R., et al.** (2012). *Concurrent training: A meta-analysis examining interference of aerobic and resistance exercises*. *J Strength Cond Res*.

14. **Tufano, J.J., Brown, L.E., Haff, G.G.** (2017). *Theoretical and practical aspects of different cluster set structures: A systematic review*. *J Strength Cond Res*.

15. **Sands, W.A., Stone, M.H., McNeal, J.R., et al.** (2006). *Flexibility enhancement with vibration: Acute and long-term*. *Med Sci Sports Exerc*.

16. **Gonzalez, A.M., Hoffman, J.R., Stout, J.R., et al.** (2016). *Intramuscular anabolic signaling and endocrine response following resistance exercise: Impact of hydration status*. *J Strength Cond Res*.

17. **Grgic, J., Schoenfeld, B.J.** (2018). *Are the hypertrophic adaptations to high and low-load resistance training muscle fiber type specific?*. *Physiol Int*.

18. **Ratamess, N.A., Alvar, B.A., Evetoch, T.K., et al.** (2009). *Progression models in resistance training for healthy adults*. *Med Sci Sports Exerc*.

19. **Folland, J.P., Williams, A.G.** (2007). *The adaptations to strength training: Morphological and neurological contributions to increased strength*. *Sports Med*.

20. **Stone, M.H., O’Bryant, H.S., Garhammer, J.** (1981). *A hypothetical model for strength training*. *J Sports Med Phys Fitness*.

21. **Stone, M.H., Sanborn, K., O'Bryant, H.S., et al.** (2003). *Maximum strength-power-performance relationships in collegiate throwers*. *J Strength Cond Res*.

22. **Hackett, D.A., Johnson, N.A., Halaki, M., Chow, C.M.** (2013). *A novel scale to assess resistance-exercise effort*. *J Sports Sci*.

23. **Goto, K., Ishii, N., Kizuka, T., Takamatsu, K.** (2004). *The impact of metabolic stress on hormonal responses and muscular adaptations*. *Med Sci Sports Exerc*.

24. **DeWeese, B.H., Sams, M.L., Serrano, A.J.** (2015). *The nature of periodization and its application for training*. *Strength Cond J*.

25. **Bartolomei, S., Hoffman, J.R., Merni, F., et al.** (2014). *Effect of different rest intervals on the strength-hypertrophy relationship*. *J Strength Cond Res*.

26. **Issurin, V.B.** (2008). *Block periodization versus traditional training theory: A review*. *J Sports Med Phys Fitness*.

27. **Fry, A.C.** (2004). *The role of resistance exercise intensity on muscle fibre adaptations*. *Sports Med*.

28. **Zatsiorsky, V.M., Kraemer, W.J.** (2006). *Science and Practice of Strength Training* (2nd ed.). Human Kinetics.
""")
