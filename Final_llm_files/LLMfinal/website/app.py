import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from flask import Flask, render_template, redirect, request, session, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import json
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
#from langchain_community.vectorstores import Chroma
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma



#  Flask setup
app = Flask(__name__)
app.secret_key = "secret"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite3'
db = SQLAlchemy(app)

#  Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))

class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    description = db.Column(db.String(300))
    price = db.Column(db.Float)

#  DB + admin
def create_admin():
    db.create_all()
    if not User.query.filter_by(email="admin").first():
        db.session.add(User(email="admin", password=generate_password_hash("admin")))
        db.session.commit()

with app.app_context():
    create_admin()

#  Routes
@app.route('/')
def index():
    products = Product.query.all()
    return render_template("index.html", products=products)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = generate_password_hash(request.form['password'])
        if User.query.filter_by(email=email).first():
            return "User already exists"
        db.session.add(User(email=email, password=password))
        db.session.commit()
        return redirect('/login')
    return render_template("signup.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(email=request.form['email']).first()
        if user and check_password_hash(user.password, request.form['password']):
            session['user'] = user.email
            return redirect('/admin' if user.email == "admin" else '/')
        return "Invalid credentials"
    return render_template("login.html")

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/')

@app.route('/admin')
def admin():
    if session.get('user') != "admin":
        return redirect('/')
    products = Product.query.all()
    return render_template("admin_panel.html", products=products)

@app.route('/add', methods=['GET', 'POST'])
def add():
    if session.get('user') != "admin":
        return redirect('/')
    if request.method == 'POST':
        db.session.add(Product(
            name=request.form['name'],
            description=request.form['description'],
            price=request.form['price']
        ))
        db.session.commit()
        return redirect('/admin')
    return render_template("add_product.html")

@app.route('/edit/<int:id>', methods=['GET', 'POST'])
def edit(id):
    if session.get('user') != "admin":
        return redirect('/')
    product = Product.query.get(id)
    if request.method == 'POST':
        product.name = request.form['name']
        product.description = request.form['description']
        product.price = request.form['price']
        db.session.commit()
        return redirect('/admin')
    return render_template("edit_product.html", product=product)

@app.route('/delete/<int:id>')
def delete(id):
    if session.get('user') != "admin":
        return redirect('/')
    db.session.delete(Product.query.get(id))
    db.session.commit()
    return redirect('/admin')

###
@app.route('/shop')
def shop():
    products = Product.query.all()
    print(products)
    return render_template("shop.html", products=products)


#@app.route('/blog')
#def blog():
 #   return render_template("blog.html")  # Static content for now

#@app.route('/cart')
#def cart():
#    return render_template("cart.html")  # Placeholder

#@app.route('/search', methods=['GET'])
#def search():
#    query = request.args.get('q')
 #   if not query:
  #      return render_template("search_results.html", results=[], query="")

   # results = Product.query.filter(Product.name.contains(query)).all()
    #return render_template("search_results.html", results=results, query=query)



#####

bart_tokenizer = AutoTokenizer.from_pretrained(r"../trained_model_fullqa")
bart_model = AutoModelForSeq2SeqLM.from_pretrained(r"../trained_model_fullqa")



#  Embeddings & Ollama setup
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=r"../chroma_index", embedding_function=embedding_model)
llm = Ollama(model="llama3")
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(search_kwargs={"k": 5}), return_source_documents=True)

#  Load dataset Q&A for similarity check
with open("../fullqa_trimmed.jsonl", "r", encoding="utf-8") as f:
    qa_data = [json.loads(line) for line in f]
dataset_questions = [entry["question"] for entry in qa_data]

similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
question_embeddings = similarity_model.encode(dataset_questions, convert_to_tensor=True)

#  Main chat endpoint
@app.route('/chat', methods=['POST'])
def chat():
    user_msg = request.json.get("message", "")
    user_embedding = similarity_model.encode(user_msg, convert_to_tensor=True)
    similarity_scores = util.cos_sim(user_embedding, question_embeddings)[0]
    top_score = float(similarity_scores.max())

    if top_score >= 0.5:
        # Use BART
        inputs = bart_tokenizer(user_msg, return_tensors="pt", truncation=True, padding=True)
        outputs = bart_model.generate(**inputs, max_new_tokens=100)
        reply = bart_tokenizer.decode(outputs[0], skip_special_tokens=True)
    else:
        # Use LLaMA
        reply = llm.invoke(user_msg)

    return jsonify({"response": reply})

if __name__ == '__main__':
    app.run(debug=True)
