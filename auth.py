import sqlite3
import hashlib

def connect_db():
    return sqlite3.connect("database.db", check_same_thread=False)

def create_table():
    conn = connect_db()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS patients(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            age INTEGER,
            gender TEXT,
            email TEXT UNIQUE,
            password TEXT
        )
    """)
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def signup(name, age, gender, email, password):
    conn = connect_db()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO patients(name, age, gender, email, password) VALUES (?, ?, ?, ?, ?)",
                  (name, age, gender, email, hash_password(password)))
        conn.commit()
        return True
    except:
        return False

def login(email, password):
    conn = connect_db()
    c = conn.cursor()
    c.execute("SELECT * FROM patients WHERE email=? AND password=?",
              (email, hash_password(password)))
    user = c.fetchone()
    return user