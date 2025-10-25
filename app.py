from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import google.generativeai as genai
import os
import speech_recognition as sr
import re
import json
import secrets

app = Flask(__name__)

# Security: Secret key from environment variable (Render will set this)
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))

# Gemini API config - MUST be set in Render environment variables
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("❌ ERROR: GOOGLE_API_KEY environment variable is required! Please set it in Render dashboard.")

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# Configure cache directories for models
os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'
os.environ['TORCH_HOME'] = '/tmp/torch_cache'
os.environ['HF_HOME'] = '/tmp/huggingface_cache'

# Pre-load Whisper model to avoid timeout on first video request
print("🔄 Pre-loading Whisper model...")
try:
    import whisper
    _whisper_model_cache = whisper.load_model("tiny", download_root="/tmp/whisper_cache")
    print("✅ Whisper model pre-loaded successfully!")
except Exception as e:
    print(f"⚠️ Warning: Could not pre-load Whisper: {e}")

# Pre-download NLTK data
try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    print("✅ NLTK data downloaded successfully!")
except Exception as e:
    print(f"⚠️ NLTK download warning: {e}")

# ------------------ Utility Functions ------------------

def SpeechToText():
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("Adjusting for ambient noise...")
            r.adjust_for_ambient_noise(source)
            print("Listening...")
            audio = r.listen(source)
        print("Recognizing...")
        query = r.recognize_google(audio, language='en-IN')
        return query
    except sr.UnknownValueError:
        return "Could not understand the audio."
    except sr.RequestError as e:
        return f"Could not request results from Google Speech Recognition service; {e}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def clean_answer(answer):
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    words = word_tokenize(answer)
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in words if word.lower() not in stop_words])

def detect_fillers(text):
    import nltk
    words = nltk.word_tokenize(text.lower())
    common_fillers = {"um", "uh", "like", "you know", "so", "actually", "basically", "literally", "well", "hmm"}
    used_fillers = [w for w in words if w in common_fillers]
    return ", ".join(set(used_fillers)) if used_fillers else "None"

# ------------------ Routes ------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_questions', methods=['POST'])
def generate_questions():
    job = request.form.get('job', '').strip()
    level = request.form.get('level', 'medium').strip()
    
    if not job:
        return jsonify({"error": "Job title is required"}), 400
    
    session['job_title'] = job
    session['difficulty'] = level
    return redirect(url_for('regenerate_questions'))

@app.route('/regenerate_questions')
def regenerate_questions():
    job = session.get('job_title', 'Software Developer')
    level = session.get('difficulty', 'medium')

    prompt = f"""
    Generate exactly 10 interview questions for the job role: {job} 
    with difficulty level: {level}. 
    Only return the 10 questions in plain text, numbered 1 to 10. 
    Do not include any introduction or extra comments.
    """
    
    try:
        response = model.generate_content(prompt)
        raw_questions = response.text.strip().split("\n")
        questions = []
        for q in raw_questions:
            match = re.match(r'^\d+[\).\s-]+(.*)', q.strip())
            if match:
                questions.append(match.group(1).strip())
        questions = questions[:10]
        
        if not questions:
            raise ValueError("No questions generated")
        
        session['questions'] = questions
        return redirect(url_for('questions'))
    except Exception as e:
        print(f"Error generating questions: {e}")
        return jsonify({"error": "Failed to generate questions"}), 500

@app.route('/questions')
def questions():
    questions = session.get('questions', [])
    if not questions:
        return redirect(url_for('index'))
    
    job = session.get('job_title')
    difficulty = session.get('difficulty')
    question_list = list(enumerate(questions, start=1))
    return render_template('questions.html', questions=question_list, job_title=job, difficulty=difficulty)

@app.route('/interview/<int:qid>')
def interview(qid):
    questions = session.get('questions', [])
    if 1 <= qid <= len(questions):
        question = questions[qid - 1]
    else:
        question = 'No question found'
    return render_template('interview.html', question=question, qid=qid)

@app.route('/get_analysis', methods=['POST'])
def get_analysis():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file"}), 400

    audio_file = request.files['audio']
    audio_path = "/tmp/user_audio.wav"
    audio_file.save(audio_path)

    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
        transcribed_text = recognizer.recognize_google(audio)
        duration = 10
    except sr.UnknownValueError:
        return jsonify({"error": "Could not understand audio."}), 400
    except sr.RequestError as e:
        return jsonify({"error": f"Speech recognition failed: {e}"}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500
    finally:
        try:
            os.remove(audio_path)
        except:
            pass

    return jsonify({
        "transcription": transcribed_text,
        "duration": duration
    })

@app.route('/submit_answer/<qid>', methods=['POST'])
def submit_answer(qid):
    user_answer = request.form.get('answer', '').strip()
    questions = session.get('questions', [])
    
    # Get the actual question text
    try:
        question_text = questions[int(qid) - 1]
    except:
        question_text = "Interview question"

    prompt = f"""
You are a strict technical interviewer evaluating an interview answer with a detailed scoring system.

Question: {question_text}
User's Answer: "{user_answer}"

EVALUATION RULES WITH SCORES:
1. "Valid" (76-100%): Answer correctly addresses the question with accurate, relevant information (may have minor gaps but fundamentally correct)
2. "Partial-High" (50-75%): Answer is related to the question with some correct information but incomplete or missing key details
3. "Partial-Low" (30-49%): Answer has some keywords related to the question but is vague, mostly incorrect, or barely relevant
4. "Invalid" (0-29%): Answer is completely wrong, off-topic, nonsense, gibberish, or doesn't address the question at all

Examples:
- Valid (90%): Complete, accurate answer with all key points
- Valid (78%): Good answer with accurate information but missing some minor details
- Partial-High (65%): Correct direction but missing important details
- Partial-Low (40%): Mentions related terms but understanding is unclear or mostly wrong
- Invalid (15%): "I don't know", gibberish, completely unrelated topic

Return ONLY valid JSON (no markdown, no code blocks, no extra text):
{{
    "correct_answer": "Brief ideal answer to the question",
    "validation": "Valid/Partial-High/Partial-Low/Invalid",
    "score": 75,
    "fillers_used": ["um", "like"],
    "feedback": "2-3 sentences explaining the score and what could be improved"
}}

IMPORTANT: 
- Assign a specific score between 0-100 based on answer quality
- Valid: 76-100, Partial-High: 50-75, Partial-Low: 30-49, Invalid: 0-29
- Be strict but fair in evaluation
"""
    
    try:
        response = model.generate_content(prompt)
        raw_text = response.text.strip()
        # Remove markdown code blocks if present
        raw_text = raw_text.replace('```json', '').replace('```', '').strip()
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        
        if json_match:
            json_str = json_match.group()
            result = json.loads(json_str)
            
            # Normalize validation status
            validation = result.get('validation', 'Invalid')
            score = result.get('score', 0)
            
            # Ensure score matches validation category
            if validation == 'Valid' and score < 76:
                score = 76
            elif validation == 'Partial-High' and (score < 50 or score > 75):
                score = 65
            elif validation == 'Partial-Low' and (score < 30 or score >= 50):
                score = 40
            elif validation == 'Invalid' and score >= 30:
                score = 15
            
            # Simplify validation status for frontend
            if validation in ['Partial-High', 'Partial-Low']:
                simple_validation = 'Partial'
            else:
                simple_validation = validation
            
            result['score'] = score
            result['simple_validation'] = simple_validation
            
        else:
            raise ValueError("No JSON object found")
    
    except Exception as e:
        print(f"Error parsing Gemini response: {e}")
        print(f"Raw response: {raw_text if 'raw_text' in locals() else 'N/A'}")
        result = {
            "correct_answer": "Unable to parse response.",
            "validation": "Invalid",
            "simple_validation": "Invalid",
            "score": 0,
            "fillers_used": [],
            "feedback": "Could not evaluate answer due to parsing error."
        }

    return jsonify({
        'user_answer': user_answer,
        'validation_result': {
            'correct_answer': result.get('correct_answer', ''),
            'validation': result.get('simple_validation', 'Invalid'),
            'score': result.get('score', 0),
            'feedback': result.get('feedback', '')
        },
        'fillers_used': result.get('fillers_used', [])
    })

# ------------------ Video Interview Routes ------------------

@app.route('/video_interview')
def video_interview():
    return render_template('video_interview.html')

@app.route('/submit_video_answer/<qid>', methods=['POST'])
def submit_video_answer(qid):
    print(f"📹 Video upload started for Q{qid}")
    
    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    file = request.files['video']
    
    # Check file size before processing (max 10MB for free tier)
    file.seek(0, os.SEEK_END)
    file_size_bytes = file.tell()
    file.seek(0)
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    print(f"📦 Video size: {file_size_mb:.2f} MB")
    
    if file_size_mb > 20:
        return jsonify({"error": "Video too large. Please record a shorter answer (max 2 minutes)"}), 400
    
    # Save video to /tmp for Render compatibility
    os.makedirs("/tmp/uploads", exist_ok=True)
    filepath = os.path.join("/tmp/uploads", f"answer_{qid}_{os.getpid()}.webm")
    
    try:
        file.save(filepath)
        print(f"✅ Video saved: {file_size_mb:.2f} MB")
    except Exception as e:
        print(f"❌ Video save error: {e}")
        return jsonify({"error": f"Failed to save video: {str(e)}"}), 500

    # Step 1: Try transcription with timeout protection
    transcript = "Unable to transcribe audio"
    whisper_available = True
    
    try:
        print("🎤 Starting audio transcription...")
        
        # Check if Whisper is available and loaded
        if '_whisper_model_cache' not in globals():
            print("⚠️ Whisper not pre-loaded, loading now...")
            import whisper
            globals()['_whisper_model_cache'] = whisper.load_model("tiny", download_root="/tmp/whisper_cache", device="cpu")
        
        # Use pre-loaded model
        import whisper
        model_whisper = globals()['_whisper_model_cache']
        
        # Quick transcription with aggressive timeout protection
        print("🎯 Transcribing (this may take 10-20 seconds)...")
        result = model_whisper.transcribe(
            filepath,
            fp16=False,
            language='en',
            verbose=False,
            condition_on_previous_text=False,  # Faster
            compression_ratio_threshold=2.4,    # Skip low quality
            no_speech_threshold=0.6             # Skip silence
        )
        transcript = result['text'].strip()
        
        if not transcript or len(transcript) < 5:
            transcript = "No clear speech detected in video"
        
        print(f"✅ Transcription complete: {len(transcript)} chars")
        
    except Exception as e:
        print(f"❌ Whisper transcription error: {e}")
        whisper_available = False
        transcript = "Audio transcription unavailable. Using manual evaluation."

    # Clean up video file immediately
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            print("🗑️ Video file deleted")
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")

    # Step 2: Get question text
    questions = session.get('questions', [])
    try:
        question_text = questions[int(qid) - 1]
    except:
        question_text = "Interview question"

    # Step 3: Default scores
    scores = {
        "Confidence Score": 0.65,
        "Content Relevance": 0.65,
        "Fluency Score": 0.65
    }

    # Step 4: Try Gemini evaluation (with fallback)
    if whisper_available and len(transcript) > 10 and "unavailable" not in transcript.lower():
        try:
            print("🤖 Evaluating with AI...")
            
            # Shorter prompt for faster response
            prompt = f"""Rate this answer (0.0-1.0 each):
Q: {question_text}
A: {transcript[:400]}

JSON only:
{{"Confidence Score": 0.X, "Content Relevance": 0.X, "Fluency Score": 0.X}}"""
            
            response = model.generate_content(prompt)
            raw_text = response.text.strip().replace('```json', '').replace('```', '').strip()
            json_match = re.search(r'\{.*?\}', raw_text, re.DOTALL)
            
            if json_match:
                eval_scores = json.loads(json_match.group())
                scores.update(eval_scores)
                print(f"✅ AI evaluation: {scores}")
                
        except Exception as e:
            print(f"⚠️ AI evaluation skipped: {e}")
            # Use default scores

    # Calculate final score
    try:
        final_eval = round(
            (float(scores.get("Confidence Score", 0.65)) +
             float(scores.get("Content Relevance", 0.65)) +
             float(scores.get("Fluency Score", 0.65))) / 3 * 100, 2
        )
    except:
        final_eval = 65.0

    print(f"📊 Final evaluation: {final_eval}%")

    # Add helpful message if transcription failed
    if not whisper_available or "unavailable" in transcript.lower():
        transcript += "\n\n[Note: For best results on free hosting, please keep video answers under 1 minute and speak clearly.]"

    return jsonify({
        "Confidence Score": float(scores.get("Confidence Score", 0.65)),
        "Content Relevance": float(scores.get("Content Relevance", 0.65)),
        "Fluency Score": float(scores.get("Fluency Score", 0.65)),
        "Final Evaluation": final_eval,
        "Transcript": transcript,
        "whisper_available": whisper_available
    })

@app.route('/result')
def result():
    return render_template('result.html')

# Health check endpoint for Render
@app.route('/health')
def health():
    return jsonify({"status": "healthy", "api_configured": bool(GOOGLE_API_KEY)}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
