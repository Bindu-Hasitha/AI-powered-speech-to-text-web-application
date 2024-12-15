from fastapi import FastAPI, File, UploadFile, HTTPException
import whisper
import uvicorn
import os

app = FastAPI()

try:
    model = whisper.load_model("large")
    print("Whisper model loaded successfully!")
except Exception as e:
    print(f"Error loading Whisper model: {e}")


@app.post("/transcribe/", summary="Transcribe an audio file")
async def transcribe_audio(file: UploadFile = File(...)):

    if not file.content_type.startswith("audio"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an audio file.")
    
    try:
        temp_file_path = os.path.abspath(f"temp_{file.filename}")
        print(f"Temporary file path: {temp_file_path}")
        
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(await file.read())
        
        if not os.path.exists(temp_file_path):
            raise HTTPException(status_code=500, detail="Temporary file was not created successfully.")
        
        print("File written successfully, starting transcription...")
        transcription_result = model.transcribe(temp_file_path)
        
        transcribed_text = transcription_result.get("text", "No text found")
        
        print("Transcription completed successfully.") 
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during transcription: {e}")
    
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print("Temporary file removed successfully.")
        else:
            print("Temporary file not found for cleanup.")
        
    return {"filename": file.filename, "transcription": transcribed_text}


@app.get("/", summary="Welcome message", description="Welcome message for the API.")
async def root():
    """Welcome endpoint to check if the API is running."""
    return {"message": "Welcome to the Audio Transcription API! Upload your audio files to transcribe them."}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
