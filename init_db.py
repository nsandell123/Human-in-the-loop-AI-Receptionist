from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=".env.local")  # <-- Add this line at the top!

import sqlite3
from datetime import datetime
import pinecone
import openai
import os
from pinecone import Pinecone, ServerlessSpec

# Connect to (or create) the database file in the current directory
conn = sqlite3.connect('requests.db')

# Create a cursor object
c = conn.cursor()

# Drop and recreate the help_requests table every time (dev only!)
c.execute('DROP TABLE IF EXISTS help_requests')

# Create the help_requests table
c.execute('''
CREATE TABLE help_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT,
    status TEXT,
    supervisor_response TEXT,
    created_at TEXT,
    answered_at TEXT
)
''')

# Seed with 5 general Q&A pairs
now = datetime.now().isoformat(timespec="seconds")
seed_data = [
    ("What are your business hours?", "resolved", "Our salon is open Monday through Saturday from 9am to 7pm, and closed on Sundays.", now, now),
    ("What services do you offer?", "resolved", "We offer haircuts, coloring, styling, blowouts, manicures, pedicures, and more. Please ask if you have a specific service in mind!", now, now),
    ("Do I need to make an appointment?", "resolved", "Appointments are recommended, but we also accept walk-ins when available.", now, now),
    ("Where are you located?", "resolved", "We are located at 123 Main Street, Anytown. Parking is available behind the building.", now, now),
    ("What is your cancellation policy?", "resolved", "We kindly ask for at least 24 hours' notice if you need to cancel or reschedule your appointment.", now, now)
]

c.executemany(
    'INSERT INTO help_requests (question, status, supervisor_response, created_at, answered_at) VALUES (?, ?, ?, ?, ?)',
    seed_data
)

# Commit changes and close the connection
conn.commit()
conn.close()

print("Database and table created and seeded successfully!")

# --- Pinecone setup and seeding ---
# Load API keys from environment variables (recommended)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_CLOUD = "aws"  # or "gcp"
PINECONE_REGION = "us-east-1"  # your region
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "faq-embeddings"
dimension = 1536  # For OpenAI ada-002 embeddings

# Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric='cosine',
        spec=ServerlessSpec(
            cloud=PINECONE_CLOUD,
            region=PINECONE_REGION
        )
    )
index = pc.Index(index_name)

# Function to get OpenAI embedding
def get_embedding(text):
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

# Upsert seed data into Pinecone
pinecone_vectors = []
for i, (question, status, answer, created_at, answered_at) in enumerate(seed_data):
    embedding = get_embedding(question)
    pinecone_vectors.append((
        str(i),  # Use string IDs for Pinecone
        embedding,
        {
            "question": question,
            "status": status,
            "supervisor_response": answer,
            "created_at": created_at,
            "answered_at": answered_at
        }
    ))
index.upsert(vectors=pinecone_vectors)
print("Seeded Pinecone with initial Q&A pairs.")
