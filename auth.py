import sqlite3
import hashlib

# =====================================================
# DATABASE CONNECTION
# =====================================================

def connect_db():

    return sqlite3.connect(
        "database.db",
        check_same_thread=False
    )

# =====================================================
# CREATE TABLE
# =====================================================

def create_table():

    conn = connect_db()

    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS patients(

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            name TEXT NOT NULL,

            age INTEGER NOT NULL,

            gender TEXT NOT NULL,

            email TEXT UNIQUE NOT NULL,

            password TEXT NOT NULL
        )
    """)

    conn.commit()

    conn.close()

# =====================================================
# PASSWORD HASHING
# =====================================================

def hash_password(password):

    return hashlib.sha256(
        password.encode()
    ).hexdigest()

# =====================================================
# SIGNUP FUNCTION
# =====================================================

def signup(name, age, gender, email, password):

    conn = connect_db()

    c = conn.cursor()

    try:

        hashed_password = hash_password(password)

        c.execute(
            """
            INSERT INTO patients
            (name, age, gender, email, password)

            VALUES (?, ?, ?, ?, ?)
            """,
            (
                name,
                age,
                gender,
                email,
                hashed_password
            )
        )

        conn.commit()

        conn.close()

        return True

    except sqlite3.IntegrityError:

        conn.close()

        return False

    except Exception as e:

        print("Signup Error:", e)

        conn.close()

        return False

# =====================================================
# LOGIN FUNCTION
# =====================================================

def login(email, password):

    conn = connect_db()

    c = conn.cursor()

    hashed_password = hash_password(password)

    c.execute(
        """
        SELECT *
        FROM patients
        WHERE email = ?
        AND password = ?
        """,
        (
            email,
            hashed_password
        )
    )

    user = c.fetchone()

    conn.close()

    return user
