from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message", "")
    result = qa_chain.invoke(user_input)
    return jsonify({"response": result["result"]})

if __name__ == '__main__':
    app.run(debug=True)
