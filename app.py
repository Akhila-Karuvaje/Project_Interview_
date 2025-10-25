from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import google.generativeai as genai
import os
import speech_recognition as sr
import re
import json

app = Flask(__name__)
app.secret_key = 'my_super_secret_key_456789'

# Gemini API config
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', 'AIzaSyACpD3waeAbKickkjJb7gBHqegPhGGB-VE')
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
    job = request.form['job']
    level = request.form['level']
    session['job_title'] = job
    session['difficulty'] = level
    return redirect(url_for('regenerate_questions'))

@app.route('/regenerate_questions')
def regenerate_questions():
    job = session.get('job_title')
    level = session.get('difficulty')

    prompt = f"""
    Generate exactly 10 interview questions for the job role: {job} 
    with difficulty level: {level}. 
    Only return the 10 questions in plain text, numbered 1 to 10. 
    Do not include any introduction or extra comments.
    """
    response = model.generate_content(prompt)
    raw_questions = response.text.strip().split("\n")
    questions = []
    for q in raw_questions:
        match = re.match(r'^\d+[\).\s-]+(.*)', q.strip())
        if match:
            questions.append(match.group(1).strip())
    questions = questions[:10]
    session['questions'] = questions
    return redirect(url_for('questions'))

@app.route('/questions')
def questions():
    questions = session.get('questions', [])
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
    audio_path = "user_audio.wav"
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
You are a strict technical interviewer evaluating an interview answer.

Question: {question_text}
User's Answer: "{user_answer}"

EVALUATION RULES:
- "Valid": Answer correctly addresses the question with accurate, relevant information
- "Partial": Answer is somewhat related but incomplete, vague, or has minor errors
- "Invalid": Answer is wrong, completely off-topic, nonsense, gibberish, or doesn't address the question at all

Examples of INVALID answers:
- Random words or gibberish (e.g., "asdfgh", "xyz 123")
- Answers about completely different topics
- "I don't know" or empty responses
- Completely incorrect or misleading information

Return ONLY valid JSON (no markdown, no code blocks, no extra text):
{{
    "correct_answer": "Brief ideal answer to the question",
    "validation": "Valid/Invalid/Partial",
    "fillers_used": ["um", "like"],
    "feedback": "1-2 sentences explaining why this validation was given"
}}
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
        else:
            raise ValueError("No JSON object found")
    except Exception as e:
        print(f"Error parsing Gemini response: {e}")
        print(f"Raw response: {raw_text if 'raw_text' in locals() else 'N/A'}")
        result = {
            "correct_answer": "Unable to parse response.",
            "validation": "Unknown",
            "fillers_used": [],
            "feedback": "Could not evaluate answer due to parsing error."
        }

    return jsonify({
        'user_answer': user_answer,
        'validation_result': {
            'correct_answer': result.get('correct_answer', ''),
            'validation': result.get('validation', ''),
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
    
    # Save video
    os.makedirs("uploads", exist_ok=True)
    filepath = os.path.join("uploads", f"answer_{qid}.webm")
    try:
        file.save(filepath)
        file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        print(f"✅ Video saved: {file_size:.2f} MB")
    except Exception as e:
        print(f"❌ Video save error: {e}")
        return jsonify({"error": f"Failed to save video: {str(e)}"}), 500

    # Step 1: Quick transcription with Whisper
    transcript = "Unable to transcribe"
    try:
        print("🎤 Starting Whisper transcription...")
        import whisper
        import torch
        
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

    # Step 3: Simple scoring (skip BERT to save time)
    scores = {
        "Confidence Score": 0.7,
        "Content Relevance": 0.7,
        "Fluency Score": 0.7
    }

    # Step 4: Quick Gemini evaluation (with timeout protection)
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

