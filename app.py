from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import google.generativeai as genai
import os
import speech_recognition as sr
import re
import json
import secrets
from werkzeug.utils import secure_filename
import tempfile

app = Flask(__name__)

# Security: Generate secret key from environment or create secure random one
app.secret_key = os.environ.get('SECRET_KEY') or secrets.token_hex(32)

# Gemini API config - CRITICAL: Must be in environment variable
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is required!")

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# Configure cache directories for models
os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'
os.environ['TORCH_HOME'] = '/tmp/torch_cache'
os.environ['HF_HOME'] = '/tmp/huggingface_cache'

# Configuration
MAX_VIDEO_SIZE_MB = 50
ALLOWED_VIDEO_EXTENSIONS = {'webm', 'mp4', 'avi'}

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

def clean_answer(answer):
    """Remove stopwords from answer text"""
    try:
        import nltk
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        words = word_tokenize(answer)
        stop_words = set(stopwords.words('english'))
        return ' '.join([word for word in words if word.lower() not in stop_words])
    except Exception as e:
        print(f"Error cleaning answer: {e}")
        return answer

def detect_fillers(text):
    """Detect filler words in text"""
    try:
        import nltk
        words = nltk.word_tokenize(text.lower())
        common_fillers = {"um", "uh", "like", "you know", "so", "actually", "basically", "literally", "well", "hmm"}
        used_fillers = [w for w in words if w in common_fillers]
        return ", ".join(set(used_fillers)) if used_fillers else "None"
    except Exception as e:
        print(f"Error detecting fillers: {e}")
        return "None"

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

def sanitize_filename(filename):
    """Sanitize filename for security"""
    return secure_filename(filename)

# ------------------ Routes ------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_questions', methods=['POST'])
def generate_questions():
    job = request.form.get('job', '').strip()
    level = request.form.get('level', 'medium').strip()
    
    # Input validation
    if not job or len(job) > 100:
        return jsonify({"error": "Invalid job title"}), 400
    
    if level not in ['easy', 'medium', 'hard']:
        level = 'medium'
    
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
        
        # Ensure we have at least some questions
        if len(questions) < 5:
            raise ValueError("Too few questions generated")
        
        session['questions'] = questions
        return redirect(url_for('questions'))
    
    except Exception as e:
        print(f"Error generating questions: {e}")
        return jsonify({"error": "Failed to generate questions. Please try again."}), 500

@app.route('/questions')
def questions():
    questions = session.get('questions', [])
    
    if not questions:
        return redirect(url_for('index'))
    
    job = session.get('job_title', 'Position')
    difficulty = session.get('difficulty', 'medium')
    question_list = list(enumerate(questions, start=1))
    return render_template('questions.html', questions=question_list, job_title=job, difficulty=difficulty)

@app.route('/interview/<int:qid>')
def interview(qid):
    questions = session.get('questions', [])
    
    if not questions or qid < 1 or qid > len(questions):
        return redirect(url_for('questions'))
    
    question = questions[qid - 1]
    return render_template('interview.html', question=question, qid=qid)

@app.route('/get_analysis', methods=['POST'])
def get_analysis():
    """Analyze audio file (legacy endpoint, can be removed if not used)"""
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file"}), 400

    audio_file = request.files['audio']
    
    # Use temporary file for security
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
        audio_path = temp_audio.name
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
        # Clean up temp file
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
    
    # Input validation
    if not user_answer or len(user_answer) > 5000:
        return jsonify({"error": "Invalid answer length"}), 400
    
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
1. "Valid" (85-100%): Answer correctly and completely addresses the question with accurate, relevant, detailed information
2. "Partial-High" (50-84%): Answer is related to the question with some correct information but incomplete or missing key details
3. "Partial-Low" (30-49%): Answer has some keywords related to the question but is vague, mostly incorrect, or barely relevant
4. "Invalid" (0-29%): Answer is completely wrong, off-topic, nonsense, gibberish, or doesn't address the question at all

Examples:
- Valid (90%): Complete, accurate answer with all key points
- Partial-High (65%): Correct direction but missing some important details
- Partial-Low (40%): Mentions related terms but understanding is unclear or mostly wrong
- Invalid (0%): "I don't know", gibberish, completely unrelated topic

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
- Valid: 85-100, Partial-High: 50-84, Partial-Low: 30-49, Invalid: 0-29
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
            if validation == 'Valid' and score < 85:
                score = 85
            elif validation == 'Partial-High' and (score < 50 or score >= 85):
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
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Validate file size (before saving)
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    if file_size > MAX_VIDEO_SIZE_MB * 1024 * 1024:
        return jsonify({"error": f"File too large. Maximum size: {MAX_VIDEO_SIZE_MB}MB"}), 400
    
    # Use temporary file for security
    with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_video:
        filepath = temp_video.name
        
    try:
        file.save(filepath)
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"✅ Video saved: {file_size_mb:.2f} MB")
    except Exception as e:
        print(f"❌ Video save error: {e}")
        return jsonify({"error": f"Failed to save video: {str(e)}"}), 500

    # Step 1: Quick transcription with Whisper
    transcript = "Unable to transcribe"
    try:
        print("🎤 Starting Whisper transcription...")
        import whisper
        
        # Force CPU and FP32 for stability
        device = "cpu"
        
        # Load model from cache
        model_whisper = whisper.load_model("tiny", download_root="/tmp/whisper_cache", device=device)
        
        # Transcribe with minimal settings for speed
        result = model_whisper.transcribe(
            filepath,
            fp16=False,  # Use FP32 on CPU
            language='en',
            verbose=False
        )
        transcript = result['text'].strip()
        print(f"✅ Transcription done: {len(transcript)} chars")
        
    except Exception as e:
        print(f"❌ Whisper error: {e}")
        transcript = f"Transcription failed: {str(e)}"

    # Clean up video file immediately to save space
    try:
        os.remove(filepath)
        print("🗑️ Video file deleted")
    except:
        pass

    # Step 2: Get question text
    questions = session.get('questions', [])
    try:
        question_text = questions[int(qid) - 1]
    except:
        question_text = "Interview question"

    # Step 3: Default scores
    scores = {
        "Confidence Score": 0.7,
        "Content Relevance": 0.7,
        "Fluency Score": 0.7
    }

    # Step 4: Gemini evaluation
    if len(transcript) > 10 and "failed" not in transcript.lower():
        try:
            print("🤖 Quick Gemini evaluation...")
            prompt = f"""
Quickly rate this interview answer with 3 scores (0.0 to 1.0):

Question: {question_text}
Answer: "{transcript[:500]}"

Return ONLY JSON:
{{"Confidence Score": 0.X, "Content Relevance": 0.X, "Fluency Score": 0.X}}
"""
            response = model.generate_content(prompt)
            raw_text = response.text.strip().replace('```json', '').replace('```', '').strip()
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            
            if json_match:
                scores = json.loads(json_match.group())
                print(f"✅ Gemini scores: {scores}")
        except Exception as e:
            print(f"⚠️ Gemini evaluation skipped: {e}")

    # Calculate final score
    try:
        final_eval = round(
            (float(scores.get("Confidence Score", 0.7)) +
             float(scores.get("Content Relevance", 0.7)) +
             float(scores.get("Fluency Score", 0.7))) / 3 * 100, 2
        )
    except:
        final_eval = 70.0

    print(f"📊 Final evaluation: {final_eval}%")

    return jsonify({
        "Confidence Score": float(scores.get("Confidence Score", 0.7)),
        "Content Relevance": float(scores.get("Content Relevance", 0.7)),
        "Fluency Score": float(scores.get("Fluency Score", 0.7)),
        "Final Evaluation": final_eval,
        "Transcript": transcript
    })

@app.route('/result')
def result():
    return render_template('result.html')

# Error handlers
@app.errorhandler(404)
def not_found(e):
    return redirect(url_for('index'))

@app.errorhandler(500)
def internal_error(e):
    print(f"Internal error: {e}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    # Set debug=False in production
    app.run(host='0.0.0.0', port=port, debug=False)
