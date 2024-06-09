import keras, keras_nlp, os
from flask import Flask, request, jsonify
# os.environ["KAGGLE_USERNAME"] = "rajafarrasdaffa"
# os.environ["KAGGLE_KEY"] = "cec21291e6d3330d892677808c9259ce"

app = Flask(__name__)

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

#load the gemma 2b model
gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("kaggle://leonardomarcellino/bangkit-medical-chatbot/keras/gemma2b")

@app.route("/generate", methods=['POST'])
def generate_text():
    data = request.json # data langsung diambil dari POST method
    # Define the template string
    template = "Instruction:\n{Patient}\n\nResponse:\n{Doctor}"
    prompt = template.format(
        Patient=data.get('Patient', ""),
        Doctor="",
    )
    sampler = keras_nlp.samplers.TopKSampler(k=1, seed=2)
    gemma_lm.compile(sampler=sampler)
    return gemma_lm.generate(prompt, max_length=256)

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8081)

