from dotenv import load_dotenv
import os
import openai
from pinecone import Pinecone
import sqlite3
from datetime import datetime

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    RoomOutputOptions,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
)
from livekit.agents.llm import function_tool
from livekit.plugins import deepgram, openai as livekit_openai, silero

load_dotenv(dotenv_path=".env.local")

# Initialize Pinecone and OpenAI clients
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "faq-embeddings"

def get_embedding(text):
    """Get OpenAI embedding for text"""
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def search_knowledge_base(user_question: str):
    """
    Search Pinecone for the best answer to a user question.
    Returns (answer, confidence) tuple.
    """
    try:
        # Get embedding for the user question
        question_embedding = get_embedding(user_question)
        
        # Search Pinecone index
        index = pc.Index(index_name)
        results = index.query(
            vector=question_embedding,
            top_k=1,  # Get the best match
            include_metadata=True
        )
        
        if results.matches:
            best_match = results.matches[0]
            confidence = best_match.score
            answer = best_match.metadata.get("supervisor_response")
            
            return answer, confidence
        else:
            return None, 0.0
            
    except Exception as e:
        print(f"Error searching knowledge base: {e}")
        return None, 0.0


class FAQAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful FAQ assistant for a salon. "
                "For every user question, call the handle_question tool using the user's actual transcript as the question. "
                "Do not make up, rephrase, or invent questions. Only use the exact question the user asked. "
                "You will get a confidence score back for each question. "
                "Keep responses brief and friendly."
            )
        )

    async def on_enter(self):
        self.session.generate_reply()

    async def escalate_to_supervisor(self, customer_question: str):
        """
        Called when the AI doesn't know how to answer a question.
        This function escalates the request to a human supervisor and logs it in the DB.
        """
        created_at = datetime.now().isoformat(timespec="seconds")
        help_request = {
            "question": customer_question,
            "status": "pending",
            "supervisor_response": None,
            "created_at": created_at,
            "answered_at": None
        }

        # Insert into SQLite DB
        try:
            conn = sqlite3.connect('requests.db')
            c = conn.cursor()
            c.execute(
                '''
                INSERT INTO help_requests (question, status, supervisor_response, created_at, answered_at)
                VALUES (?, ?, ?, ?, ?)
                ''',
                (
                    help_request["question"],
                    help_request["status"],
                    help_request["supervisor_response"],
                    help_request["created_at"],
                    help_request["answered_at"]
                )
            )
            conn.commit()
            conn.close()
            print(f"✅ Help request logged: {help_request}")
        except Exception as e:
            print(f"❌ Error logging help request: {e}")

        # Simulate texting supervisor (prints to console for now)
        print(f"📱 SUPERVISOR ALERT: Need help with '{customer_question}'")
        return "I don't have that information right now. Let me check with my supervisor and get back to you."

    @function_tool()
    async def handle_question(
        self,
        context: RunContext,
        user_question: str,
    ) -> str:
        """
        Search the knowledge base for an answer to the user's question.
        If confidence is high, return the answer. Otherwise, escalate to supervisor.
        Args:
            user_question: The question from the user - should be in user transcript (don't make up questions)
        Returns:
            The answer string, or escalates to supervisor if not confident.
        """
        answer, confidence = search_knowledge_base(user_question)
        print(f"🔍 Search result for '{user_question}': confidence={confidence}")

        if answer and confidence > 0.95:
            print(f"🔍 Answer found: {answer}")
            return answer
        else:
            print(f"⚠️ Confidence too low ({confidence}), escalating...")
            return await self.escalate_to_supervisor(user_question)


def prewarm(proc: JobProcess):
    """
    Preloads the Silero Voice Activity Detection (VAD) model into the process's userdata.
    This makes VAD available for all sessions without reloading it each time.
    """
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    """
    Main entrypoint for the agent worker. Connects to LiveKit, sets up the agent session,
    and starts the FAQAgent in the specified room.
    """
    await ctx.connect()
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        llm=livekit_openai.LLM(model="gpt-4o-mini"),
        stt=deepgram.STT(model="nova-3"),
        tts=livekit_openai.TTS(voice="ash"),
    )

    await session.start(
        agent=FAQAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )


if __name__ == "__main__":
    # Starts the agent worker using LiveKit's CLI runner, specifying the entrypoint and prewarm functions.
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
