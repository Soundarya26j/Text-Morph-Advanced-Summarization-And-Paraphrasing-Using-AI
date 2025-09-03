import os
import io
import bcrypt
import jwt
import streamlit as st
import sqlite3
import pandas as pd
import nltk
from nltk.corpus import cmudict
from nltk.tokenize import word_tokenize, sent_tokenize
import matplotlib.pyplot as plt

# ------------------- CONFIG -------------------
st.set_page_config(page_title="File Preview + Readability Analyzer", layout="wide")
SECRET_KEY = "your-secret-key"

# ------------------- DATABASE -------------------
conn = sqlite3.connect("users.db")
c = conn.cursor()
c.execute("CREATE TABLE IF NOT EXISTS users (username TEXT, password TEXT)")
conn.commit()

# ------------------- AUTH HELPERS -------------------
def create_user(username, password):
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    c.execute("INSERT INTO users (username,password) VALUES (?,?)", (username, hashed))
    conn.commit()


def login_user(username, password):
    c.execute("SELECT password FROM users WHERE username=?", (username,))
    data = c.fetchone()
    if data and bcrypt.checkpw(password.encode(), data[0]):
        token = jwt.encode({"username": username}, SECRET_KEY, algorithm="HS256")
        return token
    return None


# ------------------- READABILITY FUNCTIONS -------------------
nltk.download("punkt", quiet=True)
nltk.download("cmudict", quiet=True)
d = cmudict.dict()


def count_syllables(word):
    word = word.lower()
    if word in d:
        return [len(list(y for y in x if y[-1].isdigit())) for x in d[word]][0]
    else:
        return max(1, sum(ch in "aeiou" for ch in word))


def flesch_kincaid(text):
    words = word_tokenize(text)
    sents = sent_tokenize(text)
    syllables = sum(count_syllables(w) for w in words if w.isalpha())
    return 206.835 - 1.015 * (len(words) / len(sents)) - 84.6 * (syllables / len(words))


def gunning_fog(text):
    words = word_tokenize(text)
    sents = sent_tokenize(text)
    complex_words = [w for w in words if count_syllables(w) >= 3]
    return 0.4 * ((len(words) / len(sents)) + 100 * (len(complex_words) / len(words)))


def smog_index(text):
    sents = sent_tokenize(text)
    words = word_tokenize(text)
    complex_words = [w for w in words if count_syllables(w) >= 3]
    if len(sents) >= 30:
        return 1.0430 * (30 * (len(complex_words) / len(sents))) ** 0.5 + 3.1291
    else:
        return 0


def visualize_levels(scores):
    fig, ax = plt.subplots()
    categories = list(scores.keys())
    values = list(scores.values())
    ax.bar(categories, values, color=["green", "orange", "red"])
    ax.set_ylabel("Score")
    ax.set_title("Readability Levels")
    st.pyplot(fig)


# ------------------- STREAMLIT APP -------------------
if "token" not in st.session_state:
    st.session_state.token = None

if st.session_state.token is None:
    choice = st.sidebar.radio("Menu", ["Login", "Signup"])

    if choice == "Signup":
        st.subheader("Create Account")
        new_user = st.text_input("Username")
        new_pass = st.text_input("Password", type="password")
        if st.button("Signup"):
            create_user(new_user, new_pass)
            st.success("Account created! Please login.")

    else:
        st.subheader("Login")
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")
        if st.button("Login"):
            token = login_user(user, pwd)
            if token:
                st.session_state.token = token
                st.success("Login successful!")
            else:
                st.error("Invalid credentials")
else:
    st.sidebar.success("Logged in")
    menu = st.sidebar.radio("Dashboard", ["Home", "Readability Analyzer", "Logout"])

    if menu == "Home":
        st.title("📂 Dashboard")
        st.write("Welcome! Use the sidebar to navigate.")

    elif menu == "Readability Analyzer":
        st.title("📖 Readability Analyzer")
        text = st.text_area("Paste text here")

        if st.button("Analyze") and text.strip():
            fk = flesch_kincaid(text)
            gf = gunning_fog(text)
            smog = smog_index(text)

            st.write(f"**Flesch-Kincaid:** {fk:.2f}")
            st.write(f"**Gunning Fog:** {gf:.2f}")
            st.write(f"**SMOG Index:** {smog:.2f}")

            scores = {
                "Beginner": max(0, 100 - abs(fk)),
                "Intermediate": max(0, 100 - abs(gf * 5)),
                "Advanced": max(0, 100 - abs(smog * 7)),
            }
            visualize_levels(scores)

    elif menu == "Logout":
        st.session_state.token = None
        st.success("Logged out!")
