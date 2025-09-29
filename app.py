import os
import zipfile
import io

# === Fix OpenMP runtime error ===
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import numpy as np
import torch
import timm
import faiss
import joblib
from PIL import Image
from torchvision import transforms

# === OpenAI API ===
import base64
from io import BytesIO
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# === CONFIG ===
BASE_DIR = os.path.dirname(__file__)  # root of repo
SAVE_DIR = os.path.join(BASE_DIR, "embeddings_dino_small")
IMAGE_ROOT_DIR = os.path.join(BASE_DIR, "unsplash_images")
K = 50  # Number of similar images to retrieve

# === Load model ===
@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = timm.create_model("vit_small_patch16_224.dino", pretrained=True)
    model.eval().to(device)
    return model, device

# === Preprocessing for DINO ===
@st.cache_resource
def get_transform():
    return transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

# === Load FAISS index and paths ===
@st.cache_resource
def load_index_and_paths():
    index_path = os.path.join(SAVE_DIR, "dino_small_index.faiss")
    paths_path = os.path.join(SAVE_DIR, "dino_small_image_paths_2.npy")
    index = faiss.read_index(index_path)
    paths = np.load(paths_path, allow_pickle=True)
    return index, paths

# === Extract DINO embedding ===
def get_embedding(image: Image.Image, model, device):
    transform = get_transform()
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.forward_features(tensor)
        if features.ndim == 3:
            features = features.mean(dim=1)
        embedding = torch.nn.functional.normalize(features, dim=-1)
        return embedding.cpu().numpy().astype("float32")

# === Classifier loader ===
@st.cache_resource
def load_classifier():
    clf_path = os.path.join(SAVE_DIR, "category_classifier.pkl")
    labels_path = os.path.join(SAVE_DIR, "category_label_encoder.npy")
    clf = joblib.load(clf_path)
    classes = np.load(labels_path, allow_pickle=True)
    return clf, classes

# === Classify and optionally save ===
def classify_and_save(uploaded_file, embedding):
    clf, classes = load_classifier()
    pred = clf.predict(embedding)[0]
    category = classes[pred]  # Index directly into numpy array

    if st.button("➕ Add this image to repository"):
        category_dir = os.path.join(IMAGE_ROOT_DIR, category)
        os.makedirs(category_dir, exist_ok=True)
        save_path = os.path.join(category_dir, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"✅ Image saved to {category}/")

# === Chatbot writeup using GPT-4o-mini ===
def generate_writeup(image: Image.Image = None, query: str = None):
    if not os.environ.get("OPENAI_API_KEY"):
        return "(Stub) Please set OPENAI_API_KEY for real answers."

    try:
        system_prompt = (
            "You are an AI curator. Provide very short, museum-style descriptions of "
            "cultural or art images. Keep it to 1–2 sentences. "
            "If asked a question, answer briefly in at most one short paragraph."
        )

        # Build user content
        user_content = [{"type": "text", "text": query.strip() if query and query.strip()
                         else "Describe this artwork in a museum style (1–2 sentences)."}]

        if image is not None:
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}
            })

        # Create chat completion
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=100,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
        )

        # Extract text safely
        message_content = response.choices[0].message.content
        if isinstance(message_content, list):
            for part in message_content:
                if isinstance(part, dict) and part.get("type") == "text":
                    return part.get("text", "").strip()
                elif isinstance(part, str):
                    return part.strip()
        elif isinstance(message_content, str):
            return message_content.strip()

        return "(No response)"

    except Exception as e:
        return f"(Stub) The curator is unavailable: {e}"

# ===================================
# === UI ===
st.set_page_config(page_title="DINO Image Explorer", layout="wide")

# Tabs
tab_home, tab_fav = st.tabs(["🏠 Home", "⭐ Favorites"])

# --- Initialize session state ---
if "selected_images" not in st.session_state:
    st.session_state.selected_images = set()
if "favorites" not in st.session_state:
    st.session_state.favorites = set()

# ===================================
# === HOME TAB ===
with tab_home:
    st.title("Image Similarity Search")

    uploaded_file = st.file_uploader("Upload an image to find similar images",
                                     type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        # --- Two-column layout: uploaded image + chatbot ---
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image, caption="Uploaded Image", width=200)

        with st.spinner("Processing image..."):
            model, device = load_model()
            embedding = get_embedding(image, model, device)

        # --- Classify + save option ---
        with col1:
            classify_and_save(uploaded_file, embedding)

        # --- Chatbot write-up (GPT-4o-mini) ---
        with col2:
            st.markdown("### 🤖 Ask me")
            with st.spinner("Generating description..."):
                auto_desc = generate_writeup(image=image)
            st.info(auto_desc)

            query = st.text_input("Know more about this image")
            if query:
                with st.spinner("Thinking..."):
                    answer = generate_writeup(image=image, query=query)
                st.write(answer)

        # --- Similar images ---
        st.markdown("### 🔍 Similar Images")
        index, paths = load_index_and_paths()
        D, I = index.search(embedding, K)

        items_per_page = 10
        total_pages = (K + items_per_page - 1) // items_per_page
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)

        start = (page - 1) * items_per_page
        end = start + items_per_page
        indices = I[0][start:end]

        images_per_row = 5
        rows = [indices[i:i + images_per_row] for i in range(0, len(indices), images_per_row)]

        for row_indices in rows:
            cols = st.columns(images_per_row)
            for idx, col in zip(row_indices, cols):
                path_from_index = str(paths[idx])
                relative_path = os.path.normpath(path_from_index)
                full_img_path = os.path.join(IMAGE_ROOT_DIR, relative_path)
                if os.path.exists(full_img_path):
                    sim_img = Image.open(full_img_path).resize((224, 224))
                    with col:
                        st.image(sim_img, caption=os.path.basename(relative_path))
                        if st.checkbox("Select", key=f"sel_{relative_path}"):
                            st.session_state.selected_images.add(full_img_path)
                        if st.button("⭐", key=f"fav_{relative_path}"):
                            st.session_state.favorites.add(full_img_path)
                else:
                    col.warning(f"Missing: {full_img_path}")

        # --- Download selected images ---
        if st.session_state.selected_images:
            st.markdown("---")
            st.success(f"{len(st.session_state.selected_images)} image(s) selected.")
            if st.button("Download Selected Images as ZIP"):
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                    for path in st.session_state.selected_images:
                        if os.path.exists(path):
                            zip_file.write(path, arcname=os.path.basename(path))
                zip_buffer.seek(0)
                st.download_button(
                    label="📦 Download ZIP",
                    data=zip_buffer,
                    file_name="selected_images.zip",
                    mime="application/zip"
                )

# ===================================
# === FAVORITES TAB ===
with tab_fav:
    st.header("⭐ Your Favorites")
    if st.session_state.favorites:
        for fav in list(st.session_state.favorites):
            if os.path.exists(fav):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.image(Image.open(fav), caption=os.path.basename(fav), width=300)
                with col2:
                    if st.button("❌ Remove", key=f"remove_{fav}"):
                        st.session_state.favorites.remove(fav)
                        st.rerun()
    else:
        st.info("No favorites yet. Mark images with ⭐ in Home tab.")
