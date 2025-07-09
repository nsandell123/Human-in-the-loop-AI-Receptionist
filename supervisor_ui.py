from flask import Flask, render_template_string, request, redirect
import sqlite3
from datetime import datetime
import openai
import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv(dotenv_path=".env.local")
openai.api_key = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "faq-embeddings"

app = Flask(__name__)

DB_PATH = "requests.db"

TEMPLATE = """
<!doctype html>
<title>Supervisor UI</title>
<h2>Pending Help Requests</h2>
<table border=1>
<tr><th>ID</th><th>Question</th><th>Status</th><th>Supervisor Response</th><th>Action</th></tr>
{% for req in requests %}
<tr>
  <form method="post" action="/respond/{{ req[0] }}">
    <td>{{ req[0] }}</td>
    <td>{{ req[1] }}</td>
    <td>{{ req[2] }}</td>
    <td>
      <input name="response" value="{{ req[3] or '' }}" style="width:200px;">
    </td>
    <td>
      <button type="submit">Resolve</button>
    </td>
  </form>
</tr>
{% endfor %}
</table>
"""

def get_embedding(text):
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

@app.route("/")
def index():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, question, status, supervisor_response FROM help_requests WHERE status='pending'")
    requests_ = c.fetchall()
    conn.close()
    return render_template_string(TEMPLATE, requests=requests_)

@app.route("/respond/<int:req_id>", methods=["POST"])
def respond(req_id):
    response = request.form["response"]
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Get the question for this request
    c.execute("SELECT question FROM help_requests WHERE id=?", (req_id,))
    row = c.fetchone()
    question = row[0] if row else ""
    # Update the DB
    c.execute(
        "UPDATE help_requests SET supervisor_response=?, status='resolved', answered_at=datetime('now') WHERE id=?",
        (response, req_id)
    )
    conn.commit()
    conn.close()

    # Upsert into Pinecone
    if question and response:
        embedding = get_embedding(question)
        index = pc.Index(index_name)
        index.upsert([
            (
                str(req_id),  # Use the DB id as the Pinecone vector id
                embedding,
                {
                    "question": question,
                    "status": "resolved",
                    "supervisor_response": response,
                    "answered_at": datetime.now().isoformat(timespec="seconds")
                }
            )
        ])

    return redirect("/")

if __name__ == "__main__":
    app.run(debug=True)
