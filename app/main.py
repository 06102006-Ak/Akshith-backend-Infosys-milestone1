from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import uuid
import os

app = FastAPI(title="Multi-Agent Financial Research System")
SESSIONS_DB = {}
UPLOAD_DIR = "./uploaded_filings"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class SessionCreate(BaseModel):
    session_name: str
    description: str = None

@app.get("/")
def health_check():
    return {"status": "online", "message": "Welcome Akshi, the Financial Research API is active."}

@app.post("/api/v1/sessions/create")
def create_session(payload: SessionCreate):
    """Creates an isolated research session/workspace"""
    session_id = str(uuid.uuid4())
    SESSIONS_DB[session_id] = {
        "name": payload.session_name,
        "description": payload.description,
        "documents": []
    }
    return {"session_id": session_id, "session_details": SESSIONS_DB[session_id]}


from app.agents.document_agent import DocumentAgent

doc_agent_engine = DocumentAgent()

@app.post("/api/v1/sessions/{session_id}/upload")
async def upload_financial_document(
    session_id: str,
    company_name: str = Form(...),
    file: UploadFile = File(...)
):
    if session_id not in SESSIONS_DB:
        raise HTTPException(status_code=404, detail="Workspace session not found.")
        
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = os.path.join(UPLOAD_DIR, f"{session_id}_{company_name}_{file.filename}")
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
        
    total_chunks = doc_agent_engine.parse_and_index(
        pdf_path=file_path,
        company_name=company_name,
        session_id=session_id
    )
    SESSIONS_DB[session_id]["documents"].append({
        "company_name": company_name,
        "file_name": file.filename,
        "chunks_indexed": total_chunks
    })

    return {
        "status": "Success",
        "message": f"Document processed and securely indexed into vectors.",
        "total_chunks_created": total_chunks,
        "session_state": SESSIONS_DB[session_id]
    }

    #what i have done in this code... 
    #firstly, fastapi imported for working for backend... basemodel is for making sure that input contains str... if not neglected... 
    #initializing object app using FASTAPI() class... 
    #next i created sessions_db -> empty dictonary created to store unique ids as key and data as values...
    #uploaded_filings used to store the data... if file not exists, creates.... if there... won't give any error...
    #sessioncreate class is for checking input if it is string or not...
    #creating universal unique ids using uuid  and next storing in sessions_db... 
    # @post is for sending file to the server... 
    # later it reads uploaded file data... 
    # next it breaks into chunks and also returns no of chunks, and required details... 
