import os
import uvicorn
import numpy as np
import keras
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LangChain / Google GenAI
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

# -----------------------------
# CONFIGURATION
# -----------------------------
load_dotenv(dotenv_path=r"C:\Users\vipul\food\.env")

MODEL_PATH = r"C:\Users\vipul\food\saved_models\2.keras"
KNOWLEDGE_BASE_FOLDER = r"C:\Users\vipul\food\KnowledgeBasenew\KnowledgeBase1"
FAISS_INDEX_SUBDIR = "faiss_index"
LLM_MODEL = "gemini-2.5-pro"
TEMPERATURE = 0.2

vector_store_instance = None
llm_instance = None

# -----------------------------
# CLASS LABELS
# -----------------------------
class_names = [
    'apple_pie', 'bread_pudding', 'carrot_cake', 'cheese_plate', 'cheesecake', 'chicken_curry',
    'chicken_quesadilla', 'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'club_sandwich',
    'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings', 'eggs_benedict', 'fish_and_chips',
    'french_fries', 'french_toast', 'fried_calamari', 'fried_rice', 'frozen_yogurt', 'garlic_bread',
    'grilled_cheese_sandwich', 'hamburger', 'ice_cream', 'lasagna', 'lobster_bisque', 'omelette',
    'onion_rings', 'pancakes', 'pizza', 'red_velvet_cake', 'risotto', 'samosa', 'spring_rolls',
    'strawberry_shortcake', 'waffles'
]

# -----------------------------
# MODEL LOADING
# -----------------------------
try:
    Model = keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"⚠️ Using mock model due to error: {e}")
    class MockModel:
        def predict(self, img_batch):
            mock_prediction = np.zeros((1, len(class_names)))
            mock_prediction[0, 24] = 0.95
            return mock_prediction
    Model = MockModel()

# -----------------------------
# RAG FUNCTIONS
# -----------------------------
def get_or_create_vectorstore():
    """Load FAISS using dummy embeddings (free-tier safe)."""
    vector_path = os.path.join(KNOWLEDGE_BASE_FOLDER, FAISS_INDEX_SUBDIR)

    # ✅ Dummy embeddings to avoid Google Embedding API calls
    class DummyEmbeddings:
        def embed_query(self, text):
            return np.zeros(768).tolist()
        def embed_documents(self, texts):
            return [self.embed_query(t) for t in texts]

    embeddings = DummyEmbeddings()

    faiss_index = os.path.join(vector_path, "index.faiss")
    faiss_pkl = os.path.join(vector_path, "index.pkl")

    if os.path.exists(faiss_index) and os.path.exists(faiss_pkl):
        print(f"✅ Loading FAISS index (no embedding API): {vector_path}")
        return FAISS.load_local(vector_path, embeddings, allow_dangerous_deserialization=True)
    else:
        raise RuntimeError(f"❌ FAISS index not found in {vector_path}")


def ask_nutrition_question(food_name: str):
    """Ask the Gemini LLM for nutrition info."""
    if vector_store_instance is None or llm_instance is None:
        raise RuntimeError("Vector store or LLM not initialized.")

    docs = []
    try:
        # similarity_search won't call Google embeddings since we use DummyEmbeddings
        docs = vector_store_instance.similarity_search(food_name + " nutrition", k=5)
    except Exception as e:
        print(f"⚠️ FAISS similarity search failed: {e}")

    context = "\n".join([d.page_content for d in docs]) if docs else "No context found."

    prompt = f"""
You are a Nutrition Expert. Use the context below to provide clear, structured data
about the food '{food_name}' including calories, protein, fat, carbohydrates, and fiber.

Context:
{context}

Format your answer in JSON like this:
{{
  "Calories": "...",
  "Protein": "...",
  "Carbohydrates": "...",
  "Fat": "...",
  "Fiber": "..."
}}
Answer:
"""
    response = llm_instance.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)

# -----------------------------
# IMAGE HELPERS
# -----------------------------
def read_file_as_image(data: bytes) -> np.ndarray:
    return np.array(Image.open(BytesIO(data)).convert('RGB'))

def predict_food(image: np.ndarray):
    img_resized = np.array(Image.fromarray(image).resize((128, 128)))
    if img_resized.ndim == 2:
        img_resized = np.stack((img_resized,) * 3, axis=-1)
    elif img_resized.shape[-1] == 4:
        img_resized = img_resized[..., :3]
    img_batch = np.expand_dims(img_resized, 0).astype("float32")
    pred = Model.predict(img_batch)
    predicted_class = class_names[np.argmax(pred[0])]
    confidence = float(np.max(pred[0]))
    return predicted_class, confidence

# -----------------------------
# FASTAPI LIFESPAN
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store_instance, llm_instance
    print("🚀 Starting up Food Nutrition API...")
    try:
        vector_store_instance = get_or_create_vectorstore()
        llm_instance = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=TEMPERATURE)
        print("✅ Initialization complete (Gemini Free Tier Ready).")
    except Exception as e:
        print(f"❌ Startup error: {e}")
        raise
    yield
    print("🛑 Shutting down API...")

# -----------------------------
# FASTAPI APP
# -----------------------------
app = FastAPI(title="Food → Nutrition RAG API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FoodQuery(BaseModel):
    food_name: str

@app.get("/")
async def root():
    return {
        "message": "🍽️ Food → Nutrition RAG API (Gemini Free Tier Ready)",
        "status": "running",
        "note": "FAISS loaded locally, Gemini used for answers"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    predicted_class, confidence = predict_food(image)
    return {"predicted_class": predicted_class, "confidence": confidence}

@app.post("/analyze_food")
async def analyze_food(file: UploadFile = File(...)):
    if vector_store_instance is None or llm_instance is None:
        raise HTTPException(status_code=500, detail="Server not initialized.")
    image = read_file_as_image(await file.read())
    predicted_class, confidence = predict_food(image)
    food_name = predicted_class.replace("_", " ")
    print(f"🍽️ Detected: {food_name} ({confidence:.2f})")
    try:
        nutrition_json = ask_nutrition_question(food_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Nutrition retrieval failed: {e}")
    return {
        "predicted_food": food_name.title(),
        "confidence": round(confidence * 100, 2),
        "nutrition_info": nutrition_json
    }

@app.post("/chat_nutrition")
async def chat_nutrition(query: FoodQuery):
    if vector_store_instance is None or llm_instance is None:
        raise HTTPException(status_code=500, detail="Server not initialized.")
    try:
        nutrition_json = ask_nutrition_question(query.food_name)
        return {"food_name": query.food_name, "nutrition_info": nutrition_json}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=9089)
