import streamlit as st
import json
import os

# Cek login
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# Fungsi load dan save user
def load_users():
    if os.path.exists("users.json"):
        with open("users.json", "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open("users.json", "w") as f:
        json.dump(users, f, indent=4)

# Jika belum login
if not st.session_state["logged_in"]:
    st.title("🔐 Login / Register")

    menu = st.radio("Pilih Menu", ["Login", "Register"])

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if menu == "Register":
        name = st.text_input("Nama Lengkap")

    if st.button(menu):
        users = load_users()

        if menu == "Login":
            if email in users and users[email]["password"] == password:
                st.session_state["logged_in"] = True
                st.session_state.user = users[email]["name"]
                st.success("Login berhasil!")
                st.rerun()
            else:
                st.error("Email atau password salah.")
        elif menu == "Register":
            if email in users:
                st.warning("Email sudah terdaftar.")
            else:
                users[email] = {"name": name, "password": password}
                save_users(users)
                st.success("Registrasi berhasil! Silakan login.")
else:
    # Menu Utama setelah login
    st.sidebar.title("Menu Utama")
    pilihan = st.sidebar.radio("Pilih Halaman", ["Deteksi Jeruk", "Tentang", "Logout"])

    st.title("🍊 Aplikasi Deteksi Kematangan Buah Jeruk")
    st.write(f"Selamat datang, {st.session_state.user}!")

    if pilihan == "Deteksi Jeruk":
        import deteksi
        deteksi.run()
    elif pilihan == "Tentang":
        import about
        about.run()
    elif pilihan == "Logout":
        st.session_state.logged_in = False
        st.rerun()
