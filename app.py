from flask import Flask, request, jsonify, render_template
from hr_vector_db import hr_generate_response

app = Flask(__name__)

#login page
@app.route('/')
def login():
    return render_template('login.html')

@app.route('/search')
def index():
    return render_template('index.html')  # Your HTML frontend

@app.route('/ask', methods=['POST'])
def ask():
    role = request.json.get('role')
    question = request.json.get('question')
    print(f"Received question: {question}")
    print(f"Received role: {role}")
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    # For demonstration, we'll just return a mock response
    result = hr_generate_response(question)
    
    print( jsonify({'answer': result}))
    return jsonify({'answer': result})
    
if __name__=='__main__':
    app.run(debug=True)