from flask import request, jsonify

app = Flask(__name__)

summary_storage = {}

document_storage = {}

api_key = "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"

# Initialize LLM model
llm = ChatGroq(groq_api_key=api_key, model_name='llama3-70b-8192', temperature=0.2, top_p=0.2)

# Define the prompt
template = '''Write a very concise, well-explained, point-wise, short summary of the following text. Provide a good and user-acceptable response.
{text}
Create a section-wise summary. Also, mention what the document uploaded is aimed at doing, as in its purpose.
If applicable, display the involved parties' names as well just after the purpose. Also, at the end, mention the key findings in points.
'''

prompt = PromptTemplate(
    input_variables=['text'],
    template=template
)

@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    pdf = PdfReader(io.BytesIO(file.read()))

    text = ''
    for page in pdf.pages:
        content = page.extract_text()
        if content:
            text += content

    if not text.strip():
        return jsonify({"error": "No text extracted from the PDF"}), 400

    # Store extracted text temporarily
    document_storage[file.filename] = text

    return jsonify({"message": "Document uploaded successfully", "filename": file.filename})

@app.route('/generate_summary', methods=['POST'])
def generate_summary():
    data = request.get_json()
    filename = data.get("filename")

    if not filename:
        return jsonify({"error": "Filename is required"}), 400

    if filename not in document_storage:
        return jsonify({"error": "File not found. Please upload first."}), 404

    text = document_storage[filename]
    docs = [Document(page_content=text)]

    chain = load_summarize_chain(
        llm,
        chain_type='stuff',
        prompt=prompt,
        verbose=False
    )

    output_summary = chain.invoke(docs)
    output = output_summary['output_text']

    # Store summary
    summary_storage[filename] = output

    return jsonify({"filename": filename, "summary": output})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
